__author__ = 'teemu kanstren'

#downloads bug reports from bugresports.qt.io for analysis

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, os, datetime
import glob

fp = webdriver.FirefoxProfile()

#turn of save dialogs, save without asking to make UI scripting easier
# https://stackoverflow.com/questions/1176348/access-to-file-download-dialog-in-firefox
fp.set_preference("browser.download.folderList", 2)
fp.set_preference("browser.download.manager.showWhenStarting", False)
fp.set_preference("browser.download.dir", os.getcwd())
fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

driver = webdriver.Firefox(firefox_profile=fp)

#open the bug search page, set advance mode
driver.get("https://bugreports.qt.io/issues/?jql=")
elem = driver.find_elements_by_link_text("Advanced")[0]
elem.click()


#rename a downloaded file to given "newname".
def dl_file_rename(newname, folder_of_download):
	listing = glob.glob(folder_of_download + '/*0.csv')
	#find the latest new file in the download folder. the newest file.
	filename = max([f for f in listing], key=lambda xa: os.path.getctime(os.path.join(folder_of_download, xa)))
	max_rounds = 10
	rounds = 0

	#the general idea with this would be to wait for downloads to finish if not yet done.
	#but mostly disabled for now, typically downloads fast enoug
	while '.part' in filename:
		if rounds < max_rounds:
			rounds += 1
			print("still not finished, sleeping for "+str(rounds)+"s.")
			time.sleep(rounds)
			listing = glob.glob(folder_of_download+'/*0.csv')
			filename = max([f for f in listing], key=lambda xa: os.path.getctime(os.path.join(folder_of_download, xa)))
			continue
	if '.part' in filename:
		print("file download timed out: "+filename)
		exit(1)
	else:
		os.rename(os.path.join(folder_of_download, filename), os.path.join(folder_of_download, newname))
		print("downloaded file renamed from "+filename+" to "+newname)

#insert query string for given start and end data, download file, and rename it
def download_dates(start, end, filename):
	elem = driver.find_element_by_name("jql")
	elem.clear()
	query = 'project = QTBUG AND created >= "{start} 0:00" AND created < "{end} 00:00"'.format(start=start, end=end)
	print(query)
#	query = 'project = QTBUG AND created >= "2018/08/01 0:00" AND created < "2018/09/01 00:00"'
	elem.send_keys(query)
	elem.send_keys(Keys.RETURN)
	time.sleep(2)
	elem = driver.find_element_by_id("AJS_DROPDOWN__21")
	elem.click()
	time.sleep(2)
	elem = driver.find_element_by_id("allCsvFields")
	elem.click()
	time.sleep(2)
	elem = driver.find_element_by_id("csv-export-dialog-export-button")
	elem.click()
	time.sleep(10)
	dl_file_rename(filename, os.getcwd())


start_year = 2003
start_month = 9
#can set later start when need to re-start after error
#start_year = 2018
#start_month = 5
end_year = 2019
end_month = 3
#end_year = 2005
#end_month = 3
year = start_year
month = start_month

#the following can be used to re-download specific months data if found to be erronous
#dates = [("2009/11", "2009/12")]

#for date in dates:
#	download_dates(date[0]+"/01", date[1]+"/01", "bugs-"+date[0].replace("/", "-")+".csv")

#exit(0)

#loop to generate date query strings, enter them into service and download the reports
while True:
	#https://docs.python.org/2/library/datetime.html
	#print(datetime.date(year, month, 1).strftime("%Y/%m/%d"))
	start = datetime.date(year, month, 1).strftime("%Y/%m/%d")
	file_date = datetime.date(year, month, 1).strftime("%Y-%m")
	filename = "bugs-{file_date}.csv".format(file_date=file_date)
	if year == end_year and month == end_month:
		driver.close()
		driver.quit()
		exit(0)
	month += 1
	if month > 12:
		month = 1
		year += 1
	end = datetime.date(year, month, 1).strftime("%Y/%m/%d")
	download_dates(start, end, filename)
#	query = 'project = QTBUG AND created >= "{start} 0:00" AND created < "{end} 00:00"'.format(start=start, end=end)
#	print(query)


# https://stackoverflow.com/questions/34548041/selenium-give-file-name-when-downloading

