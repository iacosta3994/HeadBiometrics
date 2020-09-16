def remove_nested_with_idx(List_main, List_keep, list_main_idx):
    # function sistructive, making sure to copy
    list_to_modify = List_main.copy()
    # runs through each element in specific idx and compares it to specific list
    for elem_in_list in List_main:
        item = elem_in_list[list_main_idx]
        # checks item to see if value is in List_keep
        if item not in List_keep:
            list_to_modify.remove(elem_in_list)
    return list_to_modify
