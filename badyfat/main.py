import xlrd
file_path = "/Users/gong/Downloads/bodyfat.xlsx"
data = xlrd.open_workbook(file_path)
table = data.sheet_by_index(1)