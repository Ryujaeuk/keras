# 태그별 접근 방법
import bs4

html_str = """
<html>
    <body>
        <ul class="greet">
            <li>hello</li>
            <li>bye</li>
            <li>welcome</li>
        </ul>
        <ul class="reply">
            <li>ok</li>
            <li>no</li>
            <li>sure</li>
        </ul>
    </body>
</html>
"""

bs_obj = bs4.BeautifulSoup(html_str, "html.parser")
ul = bs_obj.find("ul")
print(ul)

ul = bs_obj.find("ul", {"class":"reply"}) # reply class의 내용을 추출
print(ul)
