# AI Measurement Data Bank

The AI community invests considerable effort in designing and running evaluations. Yet item-level measurement data—what system answered which item, under what conditions, and with what outcome—remain scattered across papers, repositories, leaderboards, and incompatible file formats.

The AI Measurement Data Bank is community-owned infrastructure for turning fragmented evaluation results into shared, reusable evidence. It supports systematic efforts to compare benchmarks, study validity and reliability, track capabilities over time, and design better evaluations.

- [Explore the AI Measurement Data Bank](https://aimslab.stanford.edu/measurement-db)
- [Access the released data](https://huggingface.co/datasets/aims-foundations/measurement-db)

## Contributing

Curating a benchmark is not simply a data-cleaning task. It requires reconstructing what was measured, which systems were evaluated, how the evaluation was conducted, and what each recorded outcome means. Contributors will gain firsthand experience with the structure and limitations of modern AI evaluation data. For participants in the Predictive AI Evaluation Competition, this work can also inform the development of prediction methods and expand the public evidence available for training them. For benchmark authors, curation makes their results easier to discover, compare, and reuse. For measurement researchers and practitioners, it creates a common foundation for studying the validity, reliability, and generalizability of AI evaluations. Under appropriate conditions, contributors will also be eligible for the AI Measurement Data award under the NeurIPS 2026 Predictive AI Evaluation Competition.

We welcome contributions from benchmark authors, evaluation researchers, practitioners, and students. The complete and maintained instructions are available in the [Instructions for Item-level Measurement Data Curation](link). Questions may be submitted by opening an issue. Completed curation work may be submitted for review through a pull request to this repository.


## License

To the extent that AIMS holds copyright or database rights, the original curation contributions in the AI Measurement Data Bank—including their selection, organization, standardized schema, metadata, and normalization work—are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). 

Individual benchmarks and other third-party materials are not relicensed under CC BY-SA 4.0. They retain their original licenses and terms, as identified in each benchmark's metadata. Those upstream terms govern the corresponding material.

## Citation

If you use the data curated in AI Measurement Data Bank, please cite:

```bibtex
@misc{measurementdb2026,
  title        = {The AI Measurement Data Bank},
  author       = {Truong, Nhi and Truong, Sang T. and Koyejo, Sanmi},
  year         = {2026},
  howpublished = {\url{https://aimslab.stanford.edu/measurement-db}},
  note         = {AIMS Lab, Stanford University}
}
```
