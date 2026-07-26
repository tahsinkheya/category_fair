# Category-Aware Fairness Extensions for Cornac

This repository contains the implementation accompanying the following papers:

1. **Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness**
2. **The Taming of the Bias: Measuring and Mitigating Category-Aware Gender Bias in Recommendations**

The implementation builds upon the excellent **Cornac** recommendation framework. The original recommendation models, evaluation pipeline, and several utilities are provided by the Cornac library. The category-aware fairness objectives, bias metrics, and related experimental components were implemented by **Tahsin Alamgir Kheya** on top of the Cornac framework.

## Acknowledgement

This work extends the **Cornac** recommendation library. We gratefully acknowledge the Cornac developers for providing the recommendation algorithms, training framework, and evaluation infrastructure upon which this implementation is built.

The category-aware fairness objectives, bias evaluation metrics, and associated experimental code were developed by **Tahsin Alamgir Kheya**.

## Example Usage

An example notebook, **`ml_1m_mf.ipynb`**, is provided to demonstrate the complete experimental pipeline.

The notebook performs the following steps:

1. Loads the MovieLens 1M dataset.
2. Trains the recommendation model for different values of **α (alpha)**.
3. Evaluates recommendation quality using the performance metrics provided by the Cornac evaluation framework.
4. Computes the category-aware gender bias metrics.

### Fairness Weight (α)

The parameter **α (alpha)** controls the trade-off between recommendation accuracy and fairness.

* **α = 0**: The optimization considers only the recommendation loss.
* **0 < α < 1**: The optimization balances recommendation performance and the fairness objective.
* **α = 1**: The optimization considers only the fairness objective.

By varying α, users can study the trade-off between recommendation quality and category-aware fairness.

## Category-Aware Bias Metrics

The implementation contains six category-aware bias metrics. Their names in the source code differ slightly from the terminology used in the papers.

| Code Name         | Paper Name                          |
| ----------------- | ----------------------------------- |
| `GenrePrecision`  | Category Coverage                   |
| `GenreRecall`     | Relative Category Representation    |
| `GenreMapEdited`  | Category Mean Average Precision     |
| `GenreMRR`        | Category Mean Reciprocal Rank       |
| `GenreNDCG`       | Category Discounted Cumulative Gain |
| `GenreRPrecision` | Category R-Precision                |

## References

* **Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness**
  https://dl.acm.org/doi/10.1145/3696410.3714528

* **The Taming of the Bias: Measuring and Mitigating Category-Aware Gender Bias in Recommendations**

## Contact

For questions regarding the category-aware fairness implementation or the accompanying papers, please contact the first author:

**Tahsin Alamgir Kheya**
Email: [tahsinkheya@gmail.com](mailto:tahsinkheya@gmail.com)
