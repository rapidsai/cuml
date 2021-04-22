#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
import sys
from subprocess import run
import json


class Profiler:
    def __init__(self, tmp_path='/tmp/nsys_report'):
        self.qdrep_file = tmp_path + '/report.qdrep'
        self.json_file = tmp_path + '/report.json'
        self._execute('mkdir -p ' + tmp_path)

    @staticmethod
    def _execute(command):
        res = run(command, shell=True, capture_output=True)
        if res.returncode != 0:
            raise Exception(res.stderr)

    def _nsys_profile(self, command):
        profile_command = ('nsys profile '
                           '--trace=nvtx '
                           '--force-overwrite=true '
                           '--output={qdrep_file} {command}')
        profile_command = profile_command.format(
            qdrep_file=self.qdrep_file,
            command=command)
        self._execute(profile_command)

    def _nsys_export2json(self):
        export_command = ('nsys export '
                          '--type=json '
                          '--separate-strings=true '
                          '--force-overwrite=true '
                          '--output={json_file} {qdrep_file}')
        export_command = export_command.format(
            json_file=self.json_file,
            qdrep_file=self.qdrep_file)
        self._execute(export_command)

    def _parse_json(self):
        with open(self.json_file, 'r') as json_file:
            json_content = json_file.read().replace('\n', ',')[:-1]
            json_content = '{"dict": [\n' + json_content + '\n]}'
            profile = json.loads(json_content)['dict']
            nvtx_filtered_profile = [p['NvtxEvent'] for p in profile
                                     if 'NvtxEvent' in p]
            nvtx_filtered_profile = [p for p in nvtx_filtered_profile
                                     if 'Text' in p]

            def _process_nvtx_record(record):
                new_record = {'measurement': record['Text']}
                if 'EndTimestamp' in record:
                    runtime = int(record['EndTimestamp']) -\
                              int(record['Timestamp'])
                    new_record['timestamp'] = int(record['Timestamp'])
                    new_record['runtime'] = runtime
                return new_record

            return list(map(_process_nvtx_record, nvtx_filtered_profile))

    @staticmethod
    def _display_results(results):
        filtered_results = [r for r in results if 'runtime' in r]
        filtered_results.sort(key=lambda r: r['timestamp'])
        max_length = max([len(r['measurement']) for r in filtered_results]) + 4
        for r in filtered_results:
            measurement = r['measurement']
            isutil = measurement.startswith('--')
            if isutil:
                measurement = '    ' + measurement
            measurement = measurement.ljust(max_length + 4)
            runtime = round(int(r['runtime']) / 10**9, 4)
            msg = '{measurement} : {runtime:8.4f} s'
            msg = msg.format(measurement=measurement,
                             runtime=runtime)
            if not isutil:
                msg = '\n' + msg
            print(msg)

    def profile(self, command):
        self._nsys_profile(command)
        self._nsys_export2json()
        results = self._parse_json()
        self._display_results(results)


profiler = Profiler()
profiler.profile(sys.argv[1])
