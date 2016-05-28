import scrapy
from items import NvidiablogItem

# Remove 'boto' because it causes unnecessary errors
from scrapy import optional_features
optional_features.remove('boto')

class SeekingAlphaSpider(scrapy.Spider):

    name = 'seekingalpha'
    allowed_domains = ['seekingalpha.com']
    # Generate urls based on common format
    start_urls = reversed(['http://seekingalpha.com/earnings/earnings-call-transcripts/' + str(i) for i in range(2000, 3000)])

    def parse(self, response):
        for href in response.xpath('//ul/li[@class="list-group-item article"]/h3/a/@href').extract():
            url = response.urljoin(href)
            yield scrapy.Request(url, callback=self.parse_dir_contents)

    @staticmethod
    def parse_dir_contents(response):
        for sel in response.xpath('//div[@itemprop="articleBody"]/p'):
            item = NvidiablogItem()
            item['text'] = sel.extract()
            yield item
        # Add line to mark end of transcript
        item = NvidiablogItem()
        item['text'] = 'END OF TRANSCRIPT'
        yield item
