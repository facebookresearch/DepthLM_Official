# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

from math_verify import (  # @manual=fbsource//third-party/pypi/math-verify:math-verify
    parse,
)


def delta1_metric(contents, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []
    for content, sol in zip(contents, solution):
        reward = -1.0
        # Try symbolic verification first
        try:
            answer = float(parse(content))
            reward = float(max(answer / float(sol), float(sol) / answer) < 1.25)
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == -1.0:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r"<answer>(.*?)</answer>", sol)
            ground_truth = float(
                sol_match.group(1).strip() if sol_match else sol.strip()
            )
            try:
                student_answer = float(parse(content)[0])
                reward = (
                    1.0
                    if max(student_answer / ground_truth, ground_truth / student_answer)
                    < 1.25
                    else 0.0
                )
            except Exception as e:
                print("error: ", e, "during solution parsing, content = ", content)
                reward = 0.0

        rewards.append(reward)

    return rewards


METRIC_CLASSES = {
    "delta1_metric": delta1_metric,
}
