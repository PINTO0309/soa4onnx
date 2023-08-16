# soa4onnx
**S**imple model **O**utput OP **A**dditional tools for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/soa4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/soa4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/soa4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/soa4onnx?color=2BAF2B)](https://pypi.org/project/soa4onnx/) [![CodeQL](https://github.com/PINTO0309/soa4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/soa4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170158425-7d8a87c3-b1e7-44cb-8b8f-bd3e9806f020.png" />
</p>

## 1. Setup

### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U soa4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```
$ soa4onnx -h

usage:
    soa4onnx [-h]
    -if INPUT_ONNX_FILE_PATH
    -on OUTPUT_OP_NAMES [OUTPUT_OP_NAMES ...]
    -of OUTPUT_ONNX_FILE_PATH
    [-d]
    [-n]

optional arguments:
  -h, --help
        show this help message and exit.

  -if INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
        Input onnx file path.

  -on OUTPUT_OP_NAMES [OUTPUT_OP_NAMES ...], --output_op_names OUTPUT_OP_NAMES [OUTPUT_OP_NAMES ...]
        Output name to be added to the models output OP.
        e.g.
        --output_op_names "onnx::Gather_76" "onnx::Add_89"

  -of OUTPUT_ONNX_FILE_PATH, --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path.

  -d, --do_not_type_check
        Whether not to check that input and output tensors have data types defined.'

  -n, --non_verbose
        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
>>> from soa4onnx import outputs_add
>>> help(outputs_add)

Help on function outputs_add in module soa4onnx.onnx_model_output_adder:

outputs_add(
    input_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
    output_op_names: Union[List[str], NoneType] = [],
    output_onnx_file_path: Union[str, NoneType] = '',
    do_not_type_check: Union[bool, NoneType] = False,
    non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_op_names: List[str]
        Output name to be added to the models output OP.
        If an output OP name other than one that already exists in the model is
        specified, it is ignored.
        e.g.
        output_op_names = ["onnx::Gather_76", "onnx::Add_89"]

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.
        Default: ''

    do_not_type_check: Optional[bool]
        Whether not to check that input and output tensors have data types defined.\n\
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    outputops_added_graph: onnx.ModelProto
        onnx.ModelProto with output OP added
```

## 4. CLI Execution
```bash
$ soa4onnx \
--input_onnx_file_path fusionnet_180x320.onnx \
--output_op_names "onnx::Gather_76" "onnx::Add_89" \
--output_onnx_file_path fusionnet_180x320_added.onnx
```

## 5. In-script Execution
```python
from soa4onnx import outputs_add

onnx_graph = rename(
    input_onnx_file_path="fusionnet_180x320.onnx",
    output_op_names=["onnx::Gather_76", "onnx::Add_89"],
    output_onnx_file_path="fusionnet_180x320_added.onnx",
)
```

## 6. Sample
```bash
$ soa4onnx \
--input_onnx_file_path fusionnet_180x320.onnx \
--output_op_names "onnx::Gather_76" "onnx::Add_89" \
--output_onnx_file_path fusionnet_180x320_added.onnx
```
### Before
![image](https://user-images.githubusercontent.com/33194443/169518171-aa0f7a40-18ad-4393-a409-31ac0eea24bc.png)
![image](https://user-images.githubusercontent.com/33194443/169518858-c6230f56-23c3-4925-906f-5db9f7bf8a19.png)
![image](https://user-images.githubusercontent.com/33194443/169519158-8f0e5025-a002-44f5-8856-3267110d053a.png)

### After
![image](https://user-images.githubusercontent.com/33194443/169518194-76b9306a-1bf9-4f06-ae1b-821fd84cdf02.png)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/simple-onnx-processing-tools
5. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
