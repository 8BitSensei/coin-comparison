from bs4 import BeautifulSoup
import scraper
import re
import pickle
import pprint

#===================================================
# Using Beautiful Soup and Scraper we pull image
# data from 100 pages of the British Museum catalogue
# and save said data to 'coins.pickle'.
#===================================================

page = 100
total = []
for p in range(1,page):
	raw_html = scraper.simple_get('http://www.britishmuseum.org/research/collection_online/search.aspx?searchText=coin&images=true&place=42363&from=bc&fromDate=500&to=ad&toDate=500&view=list&page='+str(p))
	html = BeautifulSoup(raw_html, 'html.parser')

	obj_list = []
	auth_list = []
	culture_list = []
	production_date_list = []
	production_place_list = []
	labels = []



	panels = html.find_all("div", class_="panel")
	for panel in panels:
		obj = panel(text=re.compile(r'Object type:'))
		if obj:
			obj_list.append(str(obj[0].parent.parent.find(text=True, recursive=False)))
		else:
			obj_list.append("None")

		obj = panel(text=re.compile(r'Authority:'))
		if obj:
			auth_list.append(str(obj[0].parent.parent.find(text=True, recursive=False)))
		else:
			auth_list.append("None")

		obj = panel(text=re.compile(r'Culture/period/dynasty:'))
		if obj:
			culture_list.append(str(obj[0].parent.parent.find(text=True, recursive=False)))
		else:
			culture_list.append("None")

		obj = panel(text=re.compile(r'Production date:'))
		if obj:
			production_date_list.append(str(obj[0].parent.parent.find(text=True, recursive=False)))
		else:
			production_date_list.append("None")

		obj = panel(text=re.compile(r'Production place:'))
		if obj:
			production_place_list.append(str(obj[0].parent.parent.find(text=True, recursive=False)))
		else:
			production_place_list.append("None")

	images = [x['src'] for x in html.findAll('img', {'class': 'landscape'})]
	print("Page: "+str(p))
	print(len(obj_list), len(auth_list), len(culture_list), len(production_date_list), len(production_place_list), len(images))

	i = 0
	for x in images:
		tmp = [obj_list[i], auth_list[i], culture_list[i], production_date_list[i], production_place_list[i], 'http://www.britishmuseum.org'+str(x)]
		total.append(tmp)
		i = i+1
	for i in range(0,len(total)):
		print(total[i][0],total[i][1],total[i][2],total[i][3],total[i][4],total[i][5])

with open('coins.pickle', 'wb') as fp:
	pickle.dump(total, fp)










