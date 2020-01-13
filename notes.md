### Slightly random notes

```python
from pysc2.lib import actions
```

`actions.FUNCTIONS` contains _all_ possible actions in the game, including race specific actions that might not be possible with the selected units.

`actions.FUNCTIONS[function_index][5]` contains the list of arguments to a function

`obs.observation` contains the entire game state, using this in its entirety is a ton of data and likely too much to use in any functional way. Creating your own features based on this data is more effective and feeding those to an algorithm is significantly more effective.

To execute your own agent:
- Have a terminal open in the directory where your script is
- `python -m pysc2.bin.agent --map map_to_load --agent filename_without_extension.Agent_class_name_in_file`
