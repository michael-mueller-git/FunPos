import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "funscript_toolbox"))

from funscript_toolbox.utils.position_annotation import position_anotation_tool_entrypoint

if __name__ == "__main__":
    position_anotation_tool_entrypoint()
