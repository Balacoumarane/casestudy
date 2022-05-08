def overlap_elements(s1, s2):
    """
    The function takes two series as input, returns common and non-overlapping elements between them

    Args:
        s1 (pd.Series or list): first series or list
        s2 (pd.Series or list): first series or list

    Returns:
        common_elements (list): common elements between two list
        different_elements (list): non-over lapping elements in two list

    """
    s1 = list(set(s1))
    s2 = list(set(s2))
    common_elements = list(set(s1).intersection(set(s2)))
    different_elements = list(set(s2) - set(s1))
    return common_elements, different_elements
