### Slightly random notes

```python
from pysc2.lib import actions
```

`actions.FUNCTIONS` contains _all_ possible actions in the game, including race specific actions that might not be possible with the selected units.

`actions.FUNCTIONS[function_index][5]` contains the list of arguments to the function
