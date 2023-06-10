DEFAULT_SEED = 42
DEFAULT_INPUT_MODEL = "t5-base"
NEW_LINE = "\n"
SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"

SOURCE_FORMAT = """{source_prefix}
{source}""".format(source_prefix=SOURCE_PREFIX, source="{source}")

TARGET_FORMAT = """{target_prefix}
{target}""".format(target_prefix=TARGET_PREFIX, target="{target}")