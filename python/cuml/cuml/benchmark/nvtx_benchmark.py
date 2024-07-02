#
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shutil
import sys
import tempfile
from subprocess import run
import json


class Profiler:
    def __init__(self, tmp_path=None):
        self.tmp_dir = tempfile.TemporaryDirectory(dir=tmp_path)
        self.nsys_file = os.path.join(self.tmp_dir.name, "report.nsys-rep")
        self.json_file = os.path.join(self.tmp_dir.name, "report.json")
        shutil.rmtree(self.tmp_dir.name)
        os.makedirs(self.tmp_dir.name, exist_ok=True)

    def __del__(self):
        self.tmp_dir.cleanup()
        self.tmp_dir = None

    @staticmethod
    def _execute(command):
        res = run(
            command,
            shell=False,
            capture_output=True,
            env=dict(os.environ, NVTX_BENCHMARK="TRUE"),
        )
        if res.returncode != 0:
            raise Exception(res.stderr)
        else:
            return res.stdout

    def _nsys_profile(self, command):
        profile_command = [
            "nsys",
            "profile",
            "--trace=nvtx",
            "--force-overwrite=true",
            "--output={nsys_file}".format(nsys_file=self.nsys_file),
        ]
        profile_command.extend(command.split(" "))
        self._execute(profile_command)

    def _nsys_export2json(self):
        export_command = [
            "nsys",
            "export",
            "--type=json",
            "--separate-strings=true",
            "--force-overwrite=true",
            "--output={json_file}".format(json_file=self.json_file),
            self.nsys_file,
        ]
        self._execute(export_command)

    def _parse_json(self):
        with open(self.json_file, "r") as json_file:
            json_content = json_file.read().replace("\n", ",")[:-1]
            json_content = '{"dict": [\n' + json_content + "\n]}"
            profile = json.loads(json_content)["dict"]

            nvtx_events = [p["NvtxEvent"] for p in profile if "NvtxEvent" in p]
            nvtx_events = [
                p for p in nvtx_events if "Text" in p and "DomainId" in p
            ]

            def get_id(attribute, lookfor, nvtx_events):
                idxs = [
                    p[attribute] for p in nvtx_events if p["Text"] == lookfor
                ]
                return idxs[0] if len(idxs) > 0 else None

            authorized_domains = {}
            for domain_name in [
                "cuml_python",
                "cuml_cpp",
                "cudf_python",
                "cudf_cpp",
            ]:
                domain_id = get_id("DomainId", domain_name, nvtx_events)
                authorized_domains[domain_id] = domain_name

            nvtx_events = [
                p
                for p in nvtx_events
                if p["DomainId"] in authorized_domains.keys()
            ]

            utils_category_id = get_id("Category", "utils", nvtx_events)

            def _process_nvtx_event(record):
                new_record = {
                    "measurement": record["Text"],
                    "start": int(record["Timestamp"]),
                }
                if "EndTimestamp" in record:
                    runtime = int(record["EndTimestamp"]) - int(
                        record["Timestamp"]
                    )
                    new_record["runtime"] = runtime
                    new_record["end"] = int(record["EndTimestamp"])
                if "DomainId" in record:
                    domain_id = record["DomainId"]
                    new_record["domain"] = authorized_domains[domain_id]
                # cuDF work and utils from cuML are categorized as utilities
                if (
                    "Category" in record
                    and record["Category"] == utils_category_id
                ) or new_record["domain"].startswith("cudf"):
                    new_record["category"] = "utils"
                else:
                    new_record["category"] = "none"
                return new_record

            return list(map(_process_nvtx_event, nvtx_events))

    @staticmethod
    def _display_results(results):
        nvtx_events = [r for r in results if "runtime" in r]
        nvtx_events.sort(key=lambda r: r["start"])
        max_length = max([len(r["measurement"]) for r in nvtx_events]) + 16

        def aggregate(records):
            agg = {}
            for r in records:
                measurement = r["measurement"]
                runtime = int(r["runtime"])
                if measurement in agg:
                    agg[measurement]["runtime"] += runtime
                else:
                    agg[measurement] = {
                        "measurement": measurement,
                        "runtime": runtime,
                        "start": r["start"],
                    }
            agg = list(agg.values())
            agg.sort(key=lambda r: r["start"])
            return agg

        def nesting_hierarchy(records):
            ends = []
            for r in records:
                ends = [e for e in ends if r["start"] < e]
                r["nesting_hierarchy"] = len(ends)
                ends.append(r["end"])
            return records

        def display(measurement, runtime):
            measurement = measurement.ljust(max_length + 4)
            runtime = round(int(runtime) / 10**9, 4)
            msg = "{measurement} : {runtime:8.4f} s"
            msg = msg.format(measurement=measurement, runtime=runtime)
            print(msg)

        while len(nvtx_events):
            record = nvtx_events[0]
            display(record["measurement"], record["runtime"])

            #  Filter events belonging to this event
            end = record["end"]
            events_to_print = [r for r in nvtx_events[1:] if r["start"] < end]

            #  Filter events and compute nesting hierarchy
            reg_events_to_print = [
                r for r in events_to_print if r["category"] != "utils"
            ]
            reg_events_to_print = nesting_hierarchy(reg_events_to_print)

            for r in reg_events_to_print:
                measurement = (
                    "    |"
                    + ("==" * r["nesting_hierarchy"])
                    + "> "
                    + r["measurement"]
                )
                display(measurement, r["runtime"])

            #  Filter utils events and aggregate them by adding up runtimes
            utils_events_to_print = [
                r for r in events_to_print if r["category"] == "utils"
            ]
            utils_events_to_print = aggregate(utils_events_to_print)

            if len(reg_events_to_print) and len(utils_events_to_print):
                print()
            if len(utils_events_to_print):
                print("    Utils summary:")
            for r in utils_events_to_print:
                display("      " + r["measurement"], r["runtime"])

            #  Remove events just displayed from the list
            nvtx_events = [r for r in nvtx_events if r["start"] >= end]
            if len(nvtx_events):
                print("\n")

    def profile(self, command):
        self._nsys_profile(command)
        self._nsys_export2json()
        results = self._parse_json()
        self._display_results(results)


if __name__ == "__main__":

    def check_version():
        stdout = Profiler._execute(["nsys", "--version"])
        full_version = stdout.decode("utf-8").split(" ")[-1]
        year, month = full_version.split(".")[:2]
        version = float(year + "." + month)
        if version < 2021.4:
            raise Exception(
                "This script requires nsys 2021.4 "
                "or later version of the tool."
            )

    check_version()
    profiler = Profiler()
    profiler.profile(sys.argv[1])
