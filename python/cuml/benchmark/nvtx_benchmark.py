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


import os
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
            filtered_profile = [p['NvtxEvent'] for p in profile
                                if 'NvtxEvent' in p]
            filtered_profile = [p for p in filtered_profile
                                if 'Text' in p and
                                'DomainId' in p]

            py_domain_id = [p['DomainId'] for p in filtered_profile
                            if p['Text'] == 'cuml_python']
            py_domain_id = py_domain_id[0] if len(py_domain_id) > 0 else None
            cpp_domain_id = [p['DomainId'] for p in filtered_profile
                             if p['Text'] == 'cuml_cpp']
            cpp_domain_id = (cpp_domain_id[0] if len(cpp_domain_id) > 0
                             else None)

            filtered_profile = [p for p in filtered_profile
                                if p['DomainId'] in [py_domain_id,
                                                     cpp_domain_id]]
            utils_category_id = [p['Category'] for p in filtered_profile
                                 if p['Text'] == 'utils']
            utils_category_id = (utils_category_id[0]
                                 if len(utils_category_id) > 0
                                 else None)

            def _process_nvtx_record(record):
                new_record = {'measurement': record['Text'],
                              'start': int(record['Timestamp'])}
                if 'EndTimestamp' in record:
                    runtime = (int(record['EndTimestamp']) -
                               int(record['Timestamp']))
                    new_record['runtime'] = runtime
                    new_record['end'] = int(record['EndTimestamp'])
                if 'DomainId' in record:
                    if record['DomainId'] == py_domain_id:
                        new_record['domain'] = 'cuml_python'
                    elif record['DomainId'] == cpp_domain_id:
                        new_record['domain'] = 'cuml_cpp'
                    else:
                        new_record['domain'] = 'none'
                if 'Category' in record and \
                   record['Category'] == utils_category_id:
                    new_record['category'] = 'utils'
                else:
                    new_record['category'] = 'none'
                new_record['is_primary'] = (
                    new_record['domain'] == 'cuml_python' and
                    new_record['category'] != 'utils'
                )
                return new_record

            return list(map(_process_nvtx_record, filtered_profile))

    @staticmethod
    def _display_results(results):
        filtered_results = [r for r in results if 'runtime' in r]
        filtered_results.sort(key=lambda r: r['start'])
        max_length = max([len(r['measurement'])
                          for r in filtered_results]) + 16

        primary_calls = [r for r in filtered_results
                         if r['is_primary']]
        other_calls = [r for r in filtered_results
                       if not r['is_primary']]

        def aggregate(calls):
            agg = {}
            for c in calls:
                measurement = c['measurement']
                runtime = int(c['runtime'])
                if measurement in agg:
                    agg[measurement]['runtime'] += runtime
                else:
                    agg[measurement] = {'measurement': measurement,
                                        'runtime': runtime,
                                        'start': c['start'],
                                        'is_primary': c['is_primary']}
            agg = list(agg.values())
            agg.sort(key=lambda r: r['start'])
            return agg

        def nesting_hierarchy(calls):
            ends = []
            for c in calls:
                ends = [e for e in ends if c['start'] < e]
                c['nesting_hierarchy'] = len(ends)
                ends.append(c['end'])
            return calls

        def display(measurement, runtime):
            measurement = measurement.ljust(max_length + 4)
            runtime = round(int(runtime) / 10**9, 4)
            msg = '{measurement} : {runtime:8.4f} s'
            msg = msg.format(measurement=measurement,
                             runtime=runtime)
            print(msg)

        for record in primary_calls:
            print()
            display(record['measurement'], record['runtime'])
            start = record['start']
            end = record['end']
            other_calls_to_print = [r for r in other_calls
                                    if r['start'] >= start and
                                    r['start'] < end]
            utils_calls_to_print = [r for r in other_calls_to_print
                                    if r['category'] == 'utils']
            utils_calls_to_print = aggregate(utils_calls_to_print)
            regular_calls_to_print = [r for r in other_calls_to_print
                                      if r['category'] != 'utils']
            regular_calls_to_print = nesting_hierarchy(regular_calls_to_print)

            for u in regular_calls_to_print:
                measurement = ('    |' + ('==' * u['nesting_hierarchy'])
                               + '> ' + u['measurement'])
                display(measurement, u['runtime'])
            if len(regular_calls_to_print):
                print()
            print('    Utils summary:')
            for u in utils_calls_to_print:
                display('      ' + u['measurement'], u['runtime'])
            other_calls = [r for r in other_calls if r['start'] >= end]

    def profile(self, command):
        os.environ['NVTX_BENCHMARK'] = 'TRUE'
        self._nsys_profile(command)
        self._nsys_export2json()
        results = self._parse_json()
        self._display_results(results)


profiler = Profiler()
profiler.profile(sys.argv[1])
