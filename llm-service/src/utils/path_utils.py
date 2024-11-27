from os import path


def get_project_root(split_at: str = "src") -> str:
    """
    Returns the path to the project root. This function splits the path to
    this module file at the directory (or directories) specified by the
    `split_at` parameter.

    #### Parameters
    - `split_at`: A string at which to split the path. This should be a path
        component immediately following the root directory.
    """
    project_root = path.dirname(__file__).split(split_at)[0]
    return project_root
