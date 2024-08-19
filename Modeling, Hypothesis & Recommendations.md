# Modeling

## 1. Hypothesis: Effect of House Rules on Customer Choice

**Hypothesis:** Strict house rules negatively impact the likelihood of a property being booked.

### Analysis Approach:

- **Categorization of House Rules:** House rules were categorized into lenient, moderate, and strict based on the language or specific rules mentioned.
- **Comparison of Booking Rates:** Booking rates for each category were calculated and compared.
- **Correlation Analysis:** A Chi-square test was performed to determine if there’s a correlation between the strictness of house rules and booking rates.

```python
# ANOVA test for house rules
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('number_of_reviews ~ C(house_rules)', data=airbnb_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
