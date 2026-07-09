"""frozen board/text 임베딩 위의 작은 contrastive alignment connector.

이 패키지는 full LLM fusion이 아니다. 두 인코더는 frozen이며 connector projection만 학습한다.
설계 근거: docs/connector_plan.md, docs/scale_v1_decision.md.
"""
