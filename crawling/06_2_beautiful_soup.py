import urllib.request
import bs4

url = "https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=103"
html = urllib.request.urlopen(url)

bs_obj = bs4.BeautifulSoup(html, "html.parser")

xx = bs_obj.find("caption",{"class":"blind"})
print(xx.text)
