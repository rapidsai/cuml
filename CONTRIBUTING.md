# Contributing to cuML

If you are interested in contributing to cuML, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/cuml/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/rapidsai/cuml/blob/master/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cuml/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/cuml/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Branches and Versions

The cuML repository has two main branches: 

1. The Master branch: it contains the last released version and only hot fixes are merged to it.  
2. Branch-x.y: it is the development branch which contains the upcoming release. All the new features should be based on this branch and Merge/Pull request should target this branch (with the exception of hotfixes).
    
### Additional details

There is a new cuML branch called 'branch-0.7', where we will be merging the PRs before the next release. The exceptions to this are the 'hotfixes' which target the issues raised by the Github users and are directly merged to the Master branch. While trying to path an issue which requires a 'hotfix', please state the intent in the PR. A few days before release the current development branch (branch-0.7) will freeze and a new branch, 'branch-0.8' will be created to continue merges.

### Steps for feature development

1. Create a new branch based on `branch-x.y` named following the format `<type>-<ext>-<name>`
    - Type: 
        - fea - For if the branch is for a new feature(s)
        - enh - For if the branch is an enhancement of an existing feature(s)
        - bug - For if the branch is for fixing a bug(s) or regression(s)
    - Name: 
        - A name to convey what is being worked on
        - Please use dashes or underscores between words as opposed to spaces.
    
2. Push whatever development you do to this branch as you would normally and when ready for review create a merge request to the appropriate branch

3. Link your code reviewer to the merge request for them to review, if unsure who should review the code, contact the relevant PIC who will either review themselves or delegate the review down to someone else:

    - cuML/C++ - @Onur Yilmaz (temporary)
    - Cython/Python - @Dante Gama Dessavre

Also, add a line to the `changelog.md` file (located in the repository root folder) with a one line description of the functionality implemented in the Pull Request. Please put the line in the adequate section: New Feature, Improvement or Bug Fix. This will be compulsory once GPUCI is fully integrated into the repository.

4. If changes are requested to the pull request from the community repeat step 3, and when the review is completed push to your public fork to update the code for the pull request

5. When the pull request is merged into the public repository master it will automatically get pulled into our internal `branch-x.y` and `nvidia` branches

6. Close the merge request on the internal repository

7. Delete your development branch unless you have plans to further extend it in the future


## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
