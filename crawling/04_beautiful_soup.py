# 속성별 접근 방법
import bs4

html_str = """
<html>
    <body>
        <ul class="ko">
            <li>
                <a href="https://www.naver.com">네이버</a>
            </li>
        </ul>
        <ul class="sns">
            <li>
                <a href="http://www.facebook.com/">페이스북</a>
            </li>
        </ul>
    </body>
</html>
"""

bs_obj = bs4.BeautifulSoup(html_str, "html.parser")
atag = bs_obj.find("a")
print(atag)

#링크만 출력
print(atag['href'])