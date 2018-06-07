# CNNs for comparing prehistoric and early historic coins

This is a project which was part of my University of Surrey Comp Sci degree. I applied a siamese CNN model to compare extracted image features of some 7000 images scraped from the [British Museum Catalogue](http://www.britishmuseum.org/research/collection_online/search.aspx).

## Preparing the Data

* To begin we need to scrape our image data by running BMScraper.py and then imageDownloader.py.
* Some images might have a black background, there's enough data to justify removing them but some may be considered valuable, if so run invert.py on the coin group.
* We apply histograms to the image data by running imageHist.py, you may choose to ignore this step or modify the histogram being used.
* Finally the data needs to be converted to a MNIST CSV format, run mnistMaker.py.

## Training the Model

* We train two models, feature and simmillarity. Feature compares groups and Simmilarity compares two images, run siamese.py.
* To display the results run loadResults.py

## Authors

* **Charlie Tizzard** - *Initial work* 

## Acknowledgments

* Colin O'Keefe for the [Web Scraper](https://realpython.com/python-web-scraping-practical-introduction/)
* Inspiration
* etc
