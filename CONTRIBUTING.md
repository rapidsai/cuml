# Contributing to cuML

If you are interested in contributing to cuML, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/cuml/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `cuml/print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
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

1. Read the project's [README.md](https://github.com/rapidsai/cuml/blob/main/README.md)
    to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cuml/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Get familar with the developer guide relevant for you:
    * For C++ developers it is available here [DEVELOPER_GUIDE.md](wiki/cpp/DEVELOPER_GUIDE.md)
    * For Python developers, a [Python DEVELOPER_GUIDE.md](wiki/python/DEVELOPER_GUIDE.md) is availabe as well.
5. Code! Make sure to update unit tests!
6. When done, [create your pull request](https://github.com/rapidsai/cuml/compare).
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a RAPIDS developer will merge your pull request.

### A note related to our CI process
After you have started a PR (refer to step 6 in the previous section), every time you do a `git push <yourRemote> <pr-branch>`, it triggers a new CI run on all the commits thus far. Even though GPUCI has mechanisms to deal with this to a certain extent, if you keep `push`ing too frequently, it might just clog our GPUCI servers and slow down every PR and conda package generation! So, please be mindful of this and try not to do many frequent pushes.

To quantify this, the average check in our CI takes between 80 and 90 minutes on our servers. The GPUCI infrastructure has limited resources, so if the servers get overwhelmed, every current active PR will not be able to correctly schedule CI.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Managing PR labels

Each PR must be labeled according to whether it is a "breaking" or "non-breaking" change (using Github labels). This is used to highlight changes that users should know about when upgrading.

For cuML, a "breaking" change is one that modifies the public, non-experimental, Python API in a
non-backward-compatible way. The C++ API does not have an expectation of backward compatibility at this
time, so changes to it are not typically considered breaking. Backward-compatible API changes to the Python
API (such as adding a new keyword argument to a function) do not need to be labeled.

Additional labels must be applied to indicate whether the change is a feature, improvement, bugfix, or documentation change. See the shared RAPIDS documentation for these labels: https://github.com/rapidsai/kb/issues/42.

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

1. `main` branch: it contains the last released version. Only hotfixes are targeted and merged into it.  
2. `branch-x.y`: it is the development branch which contains the upcoming release. All the new features should be based on this branch and Merge/Pull request should target this branch (with the exception of hotfixes).
    
### Additional details

For every new version `x.y` of cuML there is a corresponding branch called `branch-x.y`, from where new feature development starts and PRs will be targeted and merged before its release. The exceptions to this are the 'hotfixes' that target the `main` branch, which target critical issues raised by Github users and are directly merged to `main` branch, and create a new subversion of the project. While trying to patch an issue which requires a 'hotfix', please state the intent in the PR. 

For all development, your changes should be pushed into a branch (created using the naming instructions below) in your own fork of cuML and then create a pull request when the code is ready. 

A few days before releasing version `x.y` the code of the current development branch (`branch-x.y`) will be frozen and a new branch, 'branch-x+1.y' will be created to continue development.

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:
- Type: 
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name: 
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

### Building and Testing on a gpuCI image locally

Before submitting a pull request, you can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
For detailed information on usage of this script, see [here](ci/local/README.md).

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
