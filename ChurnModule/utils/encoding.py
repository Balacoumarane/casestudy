import category_encoders as ce


def count_encoding(df, cols_list):
    """
    The function does takes the column and converts the categorical value in column to numeric value based on the
    count/frequency

    Args:
        df (pd.DataFrame): data
        cols_list (list):  list of categorical columns that has be to encoded

    Returns:
         data (pd.DataFrame): encoded data based on the category count

    """
    data = df.copy()
    ce_one_hot = ce.CountEncoder(cols=cols_list).fit(data)
    data = ce_one_hot.transform(df)
    return data


def onehot_encoding(df, cols_list):
    """
    The function does takes the column and converts the categorical value in column to binary value.
    This done to replace categorical values with numerical one

    Args:
        df (pd.DataFrame): data
        cols_list (list): list of categorical columns that has be to encoded

    Returns:
        data (pd.DataFrame): encoded data

    """
    data = df.copy()
    ce_one_hot = ce.OneHotEncoder(cols=cols_list, use_cat_names=True).fit(data)
    data = ce_one_hot.transform(df)
    return data
