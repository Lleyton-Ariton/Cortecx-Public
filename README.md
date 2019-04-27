# Cortecx

Cortecx is a Python NLP library facilitating the implementation of common NLP tasks.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Cortecx.

```bash
pip install cortecx
```
## Requirements

- Numpy
- Psutil
- Python 3.6 +

For Models:
- Keras
- Tensorflow

## Usage

```python
import cortecx.constrcution.session as session
from cortecx.constrcution.materials import Parser

session.Session().start()

parser = Parser()
parser.include('YOUR PARAMS') # See Parser.possible_params

answer = parser('YOUR TEXT')
```

## Suport

If there are any issues feel free to email me at [lleyton.ariton@gmail.com]()

## Known Issues:

- Parser raises an error if input text does not end in a period ('.')

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Project Status

Work in Progress

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author
Lleyton Ariton

[lleyton.ariton@gmail.com]()
