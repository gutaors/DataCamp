# Risk and Returns: The Sharpe Ratio

An investment may make sense if we expect it to return more money than it costs. But returns are only part of the story because they are risky - there may be a range of possible outcomes. How does one compare different investments that may deliver similar results on average, but exhibit different levels of risks?

Enter William Sharpe. He introduced the reward-to-variability ratioin 1966 that soon came to be called the Sharpe Ratio. It compares the expected returns for two investment opportunities and calculates the additional return per unit of risk an investor could obtain by choosing one over the other. In particular, it looks at the difference in returns for two investments and compares the average difference to the standard deviation (as a measure of risk) of this difference. A higher Sharpe ratio means that the reward will be higher for a given amount of risk. It is common to compare a specific opportunity against a benchmark that represents an entire category of investments.

The Sharpe ratio has been one of the most popular risk/return measures in finance, not least because it's so simple to use. It also helped that Professor Sharpe won a Nobel Memorial Prize in Economics in 1990 for his work on the capital asset pricing model (CAPM).

This project calculates the Sharpe ratio for the stocks of the two tech giants Facebook and Amazon. The S&amp;P 500 that measures the performance of the 500 largest stocks in the US serves the benchmark. The calculation is documented in [notebook.ipynb](https://github.com/iDataist/Risk-and-Returns--The-Sharpe-Ratio/blob/master/notebook.ipynb). The raw data files are also stored in the repository.
