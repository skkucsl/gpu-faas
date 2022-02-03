#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Dockerfile for docker skeleton (useful for running blackbox binaries, scripts, or Python 3 actions) .
FROM bskoon/gdocker

# Upgrade and install basic Python dependencies.
RUN apt update \
  && apt -y install python3 perl jq zip git curl wget openssl ca-certificates sed openssh-client libbz2-dev g++ libc-dev \
  && update-ca-certificates \
  && pip install --upgrade pip setuptools six \
  && pip install gevent flask

ENV FLASK_PROXY_PORT 8080

RUN mkdir -p /actionProxy/owplatform
ADD actionproxy.py /actionProxy/
ADD owplatform/__init__.py /actionProxy/owplatform/
ADD owplatform/knative.py /actionProxy/owplatform/
ADD owplatform/openwhisk.py /actionProxy/owplatform/

RUN mkdir -p /action
ADD stub.sh /action/exec
RUN chmod +x /action/exec

RUN mkdir -p /root/ow_result

CMD ["/bin/bash", "-c", "cd actionProxy && python3 -u actionproxy.py"]
