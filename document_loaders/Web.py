from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.apple.com/in/shop/buy-mac?afid=p240%7Cgo~cmp-11182149775~adg-109263622693~ad-805250425600_kwd-993637289~dev-c~ext-~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-mac-mac--'

data = WebBaseLoader(url)

docs = data.load()
print(docs[0].page_content)