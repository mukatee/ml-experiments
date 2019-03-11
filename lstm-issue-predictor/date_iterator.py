__author__ = 'teemu kanstren'

#generates query strings for bugreports.qt.io

import time
import datetime
from datetime import date

start_year = 2003
start_month = 9
end_year = 2019
end_month = 1
year = start_year
month = start_month

while True:
	#https://docs.python.org/2/library/datetime.html
	#print(datetime.date(year, month, 1).strftime("%Y/%m/%d"))
	start = datetime.date(year, month, 1).strftime("%Y/%m/%d")
	if year == end_year and month == end_month:
		exit(0)
	month += 1
	if month > 12:
		month = 1
		year += 1
	end = datetime.date(year, month, 1).strftime("%Y/%m/%d")
	query = 'project = QTBUG AND created >= "{start} 0:00" AND created < "{end} 00:00"'.format(start=start, end=end)
	print(query)