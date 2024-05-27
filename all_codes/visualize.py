


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string

output=[['<s>', 0, -1, 'root'], ['\\frac', 1, 0, 'Start'], ['n', 2, 1, 'Above'], ['-', 3, 2, 'Right'], ['1', 4, 3, 'Right'], ['2', 5, 1, 'Below'], ['(', 6, 5, 'Right'], ['k', 7, 6, 'Right'], ['+', 8, 7, 'Right'], ['n', 9, 8, 'Right'], ['-', 10, 9, 'Right'], ['2', 11, 10, 'Right'], [')', 12, 11, 'Right'], ['+', 13, 1, 'Right'], ['\\frac', 14, 13, 'Right'], ['n', 15, 14, 'Above'], ['-', 16, 15, 'Right'], ['1', 17, 16, 'Right'], ['2', 18, 14, 'Below'], ['(', 19, 18, 'Right'], ['h', 20, 19, 'Right'], ['+', 21, 20, 'Right'], ['n', 22, 21, 'Right'], ['-', 23, 22, 'Right'], ['2', 24, 23, 'Right'], [')', 25, 24, 'Right']]

print(convert(1, output)) # ['<s>', '\\frac', '{', 'n', '-', '1', '2', '}', '+', '\\frac', '{', 'n', '-', '1', '2', '}', '{', 'h', '+', 'n', '-', '2', '}']  