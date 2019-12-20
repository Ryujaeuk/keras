import bs4

html_str = """
<html>
    <body>
        <ul>
            <li>hello</li>
            <li>bye</li>
            <li>welcome</li>
        </ul>
    </body>
</html>
"""
bs_obj = bs4.BeautifulSoup(html_str, "html.parser")

ul = bs_obj.find("ul")
li = ul.findAll("li")
# find = <태그>에 해당되는 데이터 하나만 출력
# findAll = <태그>에 해당되는 모든 데이터, 리스트로 출력., 데이터가 없을 경우 []만 출력

print(ul)
print(li[0]) # <li>hello</li>
print(li[1]) # <li>bye</li>
print(li[2]) # <li>welcome</li>

print(li[0].text) # hello
