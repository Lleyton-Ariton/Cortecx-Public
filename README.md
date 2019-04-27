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
import cortecx.construction.cxsession as cxsession
from cortecx.construction.materials import Parser

cxsession.Session().start()

parser = Parser()
parser.include('YOUR PARAMS') # See Parser.possible_params

answer = parser('YOUR TEXT')

```

## Example Output

Cortecx returns the processed text according to your params in easy to read and key JSON format:

Example:
```python
import cortecx.construction.cxsession as cxsession
from cortecx.construction.materials import Parser

cxsession.Session().start()

parser = Parser()
parser.include('chunks')  # 'chunks' parameter will extract noun chunks
parser.include('pos')  # 'pos' parameter will extract pos tags

answer = parser('Jimmy went to the mall.')

```

Response:
```python

>>> answer

{'analysis': [{'jimmy': {'chunk': 'B-NP', 'pos': 'NNP'}},
{'went': {'chunk': 'B-VP', 'pos': 'VBD'}},
{'to': {'chunk': 'B-PP', 'pos': 'TO'}},
{'the': {'chunk': 'B-NP', 'pos': 'DT'}},
{'mall': {'chunk': 'I-NP', 'pos': 'NN'}},
{'.': {'chunk': 'O', 'pos': '.'}}]}

```

Use the HELP class found in

```python
import cortecx.construction.tools as t

print(t.HELP().pos_codes)  # codes for the part of speech tags
print(t.HELP().chunk_codes)  # codes for the chunk tags

```

## Suport

If there are any issues feel free to email me at [lleyton.ariton@gmail.com]()

## Known Issues:

- Parser raises an error if input text does not end in a period

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
