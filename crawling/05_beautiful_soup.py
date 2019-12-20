# 네이버 글자 파싱
import urllib.request
import bs4

url = "https://www.naver.com/"
html = urllib.request.urlopen(url)

bs_obj = bs4.BeautifulSoup(html, "html.parser")

# top_right = bs_obj.find("div", {"class":"area_links"})
# first_a = top_right.find("a")
# print(first_a.text)

menu = bs_obj.find("ul", {"class":"an_l"})
menubar = menu.findAll("span", {"class":"an_txt"})

for i in range(7):
    print(menubar[i].text)