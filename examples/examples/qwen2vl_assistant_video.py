# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Ensure repo root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qwen_agent.agents import Assistant


def test():
    bot = Assistant(llm={'model': 'qwen-vl-max-latest'})

    messages = [{
         'role':
             'user',
         'content': [{
             'video': 'https://www.runoob.com/try/demo_source/mov_bbb.mp4'
         }, {
             'text': 'Describe the specific process of this video'
         }]
     }]
    # Uploading video files requires applying for permission on DashScope
    # messages = [{
    #     'role':
    #         'user',
    #     'content': [{
    #         'video': 'https://www.runoob.com/try/demo_source/mov_bbb.mp4'
    #     }, {
    #         'text': 'Describe the specific process of this video'
    #     }]
    # }]

    for rsp in bot.run(messages):
        print(rsp)


if __name__ == '__main__':
    test()
