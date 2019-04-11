#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import optparse
import datetime
import re
import pytz

from dateutil import parser
from dateutil import tz
from multiprocessing import Process, Queue
from twisted.internet import reactor

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response


class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = [
        'http://quotes.toscrape.com/tag/humor/',
    ]

    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.xpath('span/small/text()').get(),
            }

        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)


class ReutersBusinessSpider(scrapy.Spider):
    name = 'reuters'
    start_urls = [
        'https://www.reuters.com/finance'
    ]

    def parse(self, response):
        for headline in response.xpath('//div[@class="story-content"]'):
            headline_text = headline.xpath('a/h3[@class="story-title"]/text()').get().strip()
            
            # The date is particularly annoying. Recent stories only display the time in EDT
            date_text = headline.xpath('time[@class="article-time"]/span[@class="timestamp"]/text()').get().strip()
            tzinfos = {
                    'CDT': tz.gettz('US/Central'),
                    'EDT': tz.gettz('America/New York')
                }            
            datetime_text = unicode(parser.parse(date_text,
                                                 default=datetime.datetime.utcnow(),
                                                 tzinfos=tzinfos))
            yield {
                'headline': headline_text,
                'date' : datetime_text
            }

        #next_page = response.css('li.next a::attr("href")').get()
        # if next_page is not None:
        #    yield response.follow(next_page, self.parse)


class ReutersCoNewsFeedSpider(scrapy.Spider):
    name = 'reuters_co_news_feed'
    start_urls = [
        'http://feeds.reuters.com/reuters/companyNews'
    ]

    def _filter_preamble(self, headline):      
        prev_headline = headline
        new_headline = re.sub(r'^[A-Z 0-9]+\-', '', prev_headline)
        while new_headline != prev_headline:
            prev_headline = new_headline
            new_headline = re.sub(r'^[A-Z 0-9]+\-', '', prev_headline)
        return new_headline
        
    def parse(self, response):
        #inspect_response(response, self)
        for headline in response.xpath('//item'):
            headline_text = self._filter_preamble(headline.xpath('title/text()').get().strip())
            #desc_text = headline.xpath('description/text()').get().strip()
            date_text = headline.xpath('pubDate/text()').get().strip()
            datetime_utc = parser.parse(date_text, default=datetime.datetime.utcnow()).astimezone(pytz.utc)
            datetime_text = unicode(datetime_utc.isoformat().replace('+00:00', 'Z'))
            yield {
                'headline': headline_text,
                'date': datetime_text,
                #'desc': desc_text
            }


class ReutersBzNewsFeedSpider(scrapy.Spider):
    name = 'reuters_bz_news_feed'
    start_urls = [
        'http://feeds.reuters.com/reuters/businessNews'
    ]

    def parse(self, response):
        
        #inspect_response(response, self)
        for headline in response.xpath('//item'):
            headline_text = headline.xpath('title/text()').get().strip()
            #desc_text = headline.xpath('description/text()').get().strip()
            date_text = headline.xpath('pubDate/text()').get().strip()
            datetime_utc = parser.parse(date_text, default=datetime.datetime.utcnow()).astimezone(pytz.utc)
            datetime_text = unicode(datetime_utc.isoformat().replace('+00:00', 'Z'))
            yield {
                'headline': headline_text,
                'date': datetime_text,
                #'desc': desc_text
            }


def run(spider, outpath):
    def f(q):
        settings = {
            'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
            'FEED_FORMAT': 'json',
            'FEED_URI': outpath,
            'COOKIES_ENABLED': False,
            'LOG_ENABLED': True,
        }
        
        # Delete file
        if os.path.exists(outpath):
            with open(outpath, 'w') as fd:
                fd.close()

        # Stdout
        sys.stdout = open(outpath + '.stdout', "w", buffering=0)
        sys.stderr = open(outpath + '.stderr', "w", buffering=0)
        
        try:
            runner = scrapy.crawler.CrawlerRunner(settings)
            deferred = runner.crawl(spider)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            q.put(None)
        except Exception as e:
            q.put(e)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    result = q.get()
    p.join()

    if result is not None:
        raise result


def run_all(outdir):
    spiders = [ReutersCoNewsFeedSpider, ReutersBzNewsFeedSpider]
    outpaths = []
    ids = []
    for spider in spiders:
        outpath = os.path.join(outdir, 'newscrawler_%s.json' % spider.name)
        run(spider, outpath)
        outpaths.append(outpath)
        ids.append(spider.name)
    return outpaths, ids


if __name__ == "__main__":
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-o', '--out', dest='out', help='Output file to save',
                          metavar='OUT', default='tmp/newscrawler')
    (opt_args, args) = opt_parser.parse_args()

    if len(args) > 0:
        opt_parser.print_help()
        exit()

    run_all(opt_args.out)

