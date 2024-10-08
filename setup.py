# Copyright 2024 DeepMind Technologies Limited
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
# ==============================================================================

"""Setup for ALTA."""

import setuptools

REQUIRED_PACKAGES = ["absl-py", "tensorflow", "numpy", "jax",
                     "apache-beam", "scipy"]
setuptools.setup(
    name="alta",
    description="Code related to ALTA.",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    license="Apache 2.0",
    py_requires=">=3.10",
)
