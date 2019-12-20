# 네이버 뉴스 파싱
import urllib.request
import bs4

url = "https://news.naver.com/"
html = urllib.request.urlopen(url)

bs_obj = bs4.BeautifulSoup(html, "html.parser")

news = bs_obj.findAll("ul", {"class":"mlist2 no_bg"})
headline = news[0].findAll("li")

for li in headline:
    strong = li.find("strong")
    print(strong.text)