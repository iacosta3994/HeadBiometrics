def remove_nested_with_idx(List_main, List_keep, list_main_idx):
    list_to_modify = List_main.copy()
    for element in list_to_modify:
        for i in element[list_main_idx]:
            # I is the value that is being compared to List_keep removing the element if not in main list
            if not isinstance(i, List_keep):
                list_to_modify.remove(element)
    return list_to_modify
