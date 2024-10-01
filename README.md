# ALTA

This repository contains open source code related to the paper
"ALTA: Compiler-Based Analysis of Transformers".

## Installation

Clone the repository:

```shell
git clone https://github.com/google-deepmind/git
```

It is then recommended to setup a virtual environment. We provide an example
using `conda`:

```shell
conda create -n alta python=3.10
conda activate alta
```

Then install dependencies specified in `setup.py`:

```shell
pip install .
```

## Overview

The code includes the ALTA language specification
(`framework/`), symbolic interpreter
(`framework/interpreter`), and compiler (`framework/compiler`) to map ALTA programs to Transformer weights. The code also includes
a self-contained Transformer implementation (`framework/transformer`) that can be used with the compiled
weights. Finally, we include tools for extracting execution traces and using
these traces as supervision to train a MLP (`framework/traces`).

## Usage Examples

Various example programs are in the `examples/` directory. The unit tests
show examples of running these programs using the symbolic interpreter and by
compiling them to Transformer weights.

Here is an example of running the unit tests for the parity program:

```
python -m examples.parity_test
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
