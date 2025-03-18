# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.0](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.2.0) - 2025-02-05

<small>[Compare with first commit](https://github.com/mathysgrapotte/stimulus-py/compare/fdd22770efcb04c2b8cc3e8772ea677afec72395...0.2.0)</small>

### Features

- made optional params Optional in Pydantic definition ([01b060d](https://github.com/mathysgrapotte/stimulus-py/commit/01b060d57e4eb8d690fee69a2f591e4d47f5ea92) by mgrapotte).

### Bug Fixes

- added to gitignore. ([c2f6c03](https://github.com/mathysgrapotte/stimulus-py/commit/c2f6c037245a602201a4a6e3848ba509356c6809) by mgrapotte).
- sometimes tuning reports no best trial found ([daf30f2](https://github.com/mathysgrapotte/stimulus-py/commit/daf30f2002ff8fd20bac8947b5b370c2bc9c9736) by mgrapotte).
- added extra tests. ([503048a](https://github.com/mathysgrapotte/stimulus-py/commit/503048a794b5ac0de01ff747e1e0b2a52125c585) by mgrapotte).
- tune should output results ([e0ee5a3](https://github.com/mathysgrapotte/stimulus-py/commit/e0ee5a31052c90a1d6c04b0fe9d94f90c8b86dac) by mgrapotte).
- added error checking for checking that output of tune is not None ([d9e1bd5](https://github.com/mathysgrapotte/stimulus-py/commit/d9e1bd55cef5655d7c4938cc26fb2dbd2c778589) by mgrapotte).
- fixed bad arg definition. ([ddcc969](https://github.com/mathysgrapotte/stimulus-py/commit/ddcc96923764384a8e68f5dfd6a5adacf306d0f4) by mgrapotte).
- fix documentation error in check_model ([7af84d0](https://github.com/mathysgrapotte/stimulus-py/commit/7af84d05d173d1587a935a9c78b30590758c813d) by mgrapotte).
- fixed pytest not closing files and moved ray init outside of check_model. ([15a92de](https://github.com/mathysgrapotte/stimulus-py/commit/15a92de17e2cb4d432e233e7ad59a91893a496be) by mgrapotte).
- run make format ([f8dcaf6](https://github.com/mathysgrapotte/stimulus-py/commit/f8dcaf6391cf2b2f179c4668b5003a0c73ee3a35) by mgrapotte).
- ran make format ([58cc43c](https://github.com/mathysgrapotte/stimulus-py/commit/58cc43c3911e84f5be4b55c6d4eeb28af1ac4ebe) by mgrapotte).
- added comments and ran make format ([200e30a](https://github.com/mathysgrapotte/stimulus-py/commit/200e30afd00bbcd9ca142c4dba2dd05b3af67716) by mgrapotte).
- added debug section for checking tensor shapes and make format ([748b887](https://github.com/mathysgrapotte/stimulus-py/commit/748b887c23159c3c893a125294df597f1429271a) by mgrapotte).
- fix linting issues ([6ea4bb0](https://github.com/mathysgrapotte/stimulus-py/commit/6ea4bb02cdf84a508650828f4e5780857e896b8d) by mgrapotte).
- run make format ([4256e88](https://github.com/mathysgrapotte/stimulus-py/commit/4256e88d73a54490b82ede27848f0c5a404ce6f6) by mgrapotte).
- correct tensor shapes. - stack during forward was not well executed - target shape was incorrect ([b37f099](https://github.com/mathysgrapotte/stimulus-py/commit/b37f099561e929f88d657e4610dc76461549fb60) by mgrapotte).
- make format ([189aab7](https://github.com/mathysgrapotte/stimulus-py/commit/189aab700df7779b4db5fe3fe48e9cac620909a2) by mgrapotte).
- current implementation was considering everything as a slice. ([539089d](https://github.com/mathysgrapotte/stimulus-py/commit/539089db8aad2558b77906b4e4546d4c2ca44252) by mgrapotte).
- pass data refs through config instead of function def ([fe76db1](https://github.com/mathysgrapotte/stimulus-py/commit/fe76db1da20b7506ee833a38fc45bdd36a51a05a) by mgrapotte).
- replace bad model_param keyword with network_params in config ([476b322](https://github.com/mathysgrapotte/stimulus-py/commit/476b322ff6c667bfafcd56aa502695095dd80c8c) by mgrapotte).
- fixed arg issue ([86cd582](https://github.com/mathysgrapotte/stimulus-py/commit/86cd58208f72b460c6b8da283a4590d15c7fd35b) by mgrapotte).
- fixed data init to be within the trainable setup, this prevents from passing data through ray object store ([3e83e8b](https://github.com/mathysgrapotte/stimulus-py/commit/3e83e8b928c281f8dc27a91b6ddf1f11ce295ef9) by mgrapotte).
- fix issue where raytune was not shutting down properly ([74c1944](https://github.com/mathysgrapotte/stimulus-py/commit/74c194455ba4ec70f4bdf08d78258a8437072df0) by mgrapotte).
- make format ([789591f](https://github.com/mathysgrapotte/stimulus-py/commit/789591f64beb83c560a0b7cf42095b1acc84bc0f) by mgrapotte).
- add __init__.py to make linter happy ([01bf0bf](https://github.com/mathysgrapotte/stimulus-py/commit/01bf0bfd4f9ba887f836ce23423477b992d0d840) by mgrapotte).
- fixed issues in raytune_learner ([45e3292](https://github.com/mathysgrapotte/stimulus-py/commit/45e3292c80bea0f5b501ce5058ffa7726deada3a) by mgrapotte).
- fixed import error. ([15f809f](https://github.com/mathysgrapotte/stimulus-py/commit/15f809f762d8b335c01b25cba06e5e90cfda2212) by mgrapotte).
- main was calling args.json instead of args.yaml ([b5265b5](https://github.com/mathysgrapotte/stimulus-py/commit/b5265b554e9ded72848c676640de6c1eff8fc125) by mgrapotte).
- fix imports in raytune_learner ([5bf0481](https://github.com/mathysgrapotte/stimulus-py/commit/5bf0481564b6043a04957eceba843b6c08d60cea) by mgrapotte).
- resolve merge conflicts ([5e6faa0](https://github.com/mathysgrapotte/stimulus-py/commit/5e6faa0212fde747151ff4721f49a537d3fcaee4) by mgrapotte).
- added mode field to custom parameter class ([8f8cd06](https://github.com/mathysgrapotte/stimulus-py/commit/8f8cd06f2e4323da835d44c370c759d84daa797f) by mgrapotte).
- added arbitrary_type support for Domain ([72e692e](https://github.com/mathysgrapotte/stimulus-py/commit/72e692e5a7f629752f0dddd8da9d36ee0be5149d) by mgrapotte).
- changed order of validator to output better error messages ([39f1d0c](https://github.com/mathysgrapotte/stimulus-py/commit/39f1d0cc450e49bf6e1ad0af6a24b158a511f7ec) by mgrapotte).
- modified gpu test config to accomodate for new Pydantic format ([97158b5](https://github.com/mathysgrapotte/stimulus-py/commit/97158b5f55c59f1bda11895a81586bff8ba86fdf) by mgrapotte).
- fixed linting by adding punctuation to main docstring ([c075a12](https://github.com/mathysgrapotte/stimulus-py/commit/c075a120ed13ca2ea663f967788fcbd4ffb17895) by mgrapotte).
- model_ is a pydantic protected namespace, replaced by network_ ([d94df0a](https://github.com/mathysgrapotte/stimulus-py/commit/d94df0acfa155956fd169dae8d2104b035efeba3) by mgrapotte).

### Code Refactoring

- removed analysis default as it was outdated ([3833486](https://github.com/mathysgrapotte/stimulus-py/commit/3833486bbc71e84228e50ae54e4a4e230f48cba5) by mgrapotte).
- use ray grid instead of dict. ([6f6471e](https://github.com/mathysgrapotte/stimulus-py/commit/6f6471e51147d19882ee6b4845077ad984273daf) by mgrapotte).
- added error detection for tuning parsing. ([31f833c](https://github.com/mathysgrapotte/stimulus-py/commit/31f833c9b4f90daad11e56280403e556bfe84cf4) by mgrapotte).
- refactored tuning cli to comply with current implementations. ([8d627ed](https://github.com/mathysgrapotte/stimulus-py/commit/8d627ed7e0cb56e68c654f6ba115233e23ed1529) by mgrapotte).
- refactored check_model cli ([ac60446](https://github.com/mathysgrapotte/stimulus-py/commit/ac60446302fbca5e1fdfb5223c5d4cde10b116b4) by mgrapotte).
- removed unused launch utils. ([4e4519d](https://github.com/mathysgrapotte/stimulus-py/commit/4e4519dea335b2499825215b94473e82c24e7b28) by mgrapotte).
- fixed import to use refactored classes and removed unused flag with current paradigm ([f014373](https://github.com/mathysgrapotte/stimulus-py/commit/f014373e42ec6cfc4a8f59fd823ede85e6432d8b) by mgrapotte).
- removed check_ressources function ([088f85c](https://github.com/mathysgrapotte/stimulus-py/commit/088f85c020b1edc5ced00844c7a710978f66ecac) by mgrapotte).
- explicit declaration of RunConfig ([d2ceca8](https://github.com/mathysgrapotte/stimulus-py/commit/d2ceca86a990c58ef2cd18011c90ecaa0c5dfc90) by mgrapotte).
- refactored TuneConfig creation ([47061ce](https://github.com/mathysgrapotte/stimulus-py/commit/47061ce158b031ad4488bbb7a9ad1240b5fea399) by mgrapotte).
- now takes as input the seed instead of loading it from the config ([270f068](https://github.com/mathysgrapotte/stimulus-py/commit/270f06835833f4468e8e9220f06ed73dbcdc4205) by mgrapotte).
- now takes as input the model config directly instead of path ([1868e71](https://github.com/mathysgrapotte/stimulus-py/commit/1868e71d0d1235aee72955dce82b0da0140f7f56) by mgrapotte).
- run make format ([07ca1c6](https://github.com/mathysgrapotte/stimulus-py/commit/07ca1c62c0efdb8c9b8bbc9e1056d3e43ffaa277) by mgrapotte).
- removed ressource allocation specifics since ray cluster will be initialized outside of the python script ([7fb60d0](https://github.com/mathysgrapotte/stimulus-py/commit/7fb60d092c2e194ac977799a27d4d3c650d7da7e) by mgrapotte).
- modified model_schema to output pydantic class instead of dumped model ([d0f6d07](https://github.com/mathysgrapotte/stimulus-py/commit/d0f6d0714d3ae2d20f642b5c0414099c06e224fb) by mgrapotte).
- YamlConfigLoader now fully depends on pydantic class ([4931f3b](https://github.com/mathysgrapotte/stimulus-py/commit/4931f3bd43ed282aff114f4cc7a8bdd147b944b6) by mgrapotte).
- class YamlRayConfigLoader should properly use Pydantic classes ([7241179](https://github.com/mathysgrapotte/stimulus-py/commit/724117948d53264b807998aec6dca3c8cd6d6844) by mgrapotte). todo: Pydantic class does not yet use TunableParameters, YamlRayConfigLoader should be adapted for this (mostly the convert to ray method)
- moved validation from space_selector to pydantic ([5bc0008](https://github.com/mathysgrapotte/stimulus-py/commit/5bc00082d1bf2f0062bade1ea51c9ac0cba1683f) by mgrapotte).
- adding a pydantic class for dealing with tunable parameters ([9213522](https://github.com/mathysgrapotte/stimulus-py/commit/9213522c79fbcda24a3db18f6c7cb1418c44fbb0) by mgrapotte).
- reverted changes for adding raytune objects in yaml config. ([660fdc3](https://github.com/mathysgrapotte/stimulus-py/commit/660fdc39b1a7a0baf80933101727feab506cc00c) by mgrapotte).
- added Pydantic classes for parsing model yaml ([6ee91ea](https://github.com/mathysgrapotte/stimulus-py/commit/6ee91ea9962a9825d973fd5a42d962130e7770d8) by mgrapotte).
- have model yaml take ray search space directly ([1ee47aa](https://github.com/mathysgrapotte/stimulus-py/commit/1ee47aa086aef20c7a7553ba6bef8190810c7a9a) by mgrapotte).
- improve TransformLoader initialization and organization ([94ca92b](https://github.com/mathysgrapotte/stimulus-py/commit/94ca92bd8e5386b62e80192a19a0afb8efaea5f9) by mgrapotte).
- implement initial manager classes for dataset handling ([f27101d](https://github.com/mathysgrapotte/stimulus-py/commit/f27101decb4a57fc20bf6b2f7dc1532013b29203) by mgrapotte).

## [0.2.1](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.2.1) - 2025-02-05

<small>[Compare with 0.2.0](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.0...0.2.1)</small>

## [0.2.2](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.2.2) - 2025-02-05 

<small>[Compare with 0.2.1](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.1...0.2.2)</small>

### Bug Fixes

- remove deprecated tests from analysis types. ([1dc4ed9](https://github.com/mathysgrapotte/stimulus-py/commit/1dc4ed96326f04adf8b4b5d7d7e74bd62e71953d) by mgrapotte).
- removed deprecated types from analysis in __init__.py. ([7a7390f](https://github.com/mathysgrapotte/stimulus-py/commit/7a7390ff91b83a1974726dd8da9f26f81932fa18) by mgrapotte).
- added split-yaml removed deprecated split json and run analysis-default. ([ddd9c9f](https://github.com/mathysgrapotte/stimulus-py/commit/ddd9c9fdd4ccf5445682008d271c9f4d648e6f22) by mgrapotte).

### Code Refactoring

- removed analysis cli since it is deprecated. ([e2f44cf](https://github.com/mathysgrapotte/stimulus-py/commit/e2f44cf13e24111a4176efbcca2bd608da7e6f46) by mgrapotte). 


## [0.2.3](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.2.3) - 2025-02-07

<small>[Compare with 0.2.2](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.2...0.2.3)</small>

### Bug Fixes

- src.stimulus.data -> stimulus.data. ([b04546c](https://github.com/mathysgrapotte/stimulus-py/commit/b04546c2c9bcc1f6218c31d8f7132e57ea1dab91) by mgrapotte).
- src.stimulus.utils -> stimulus.utils. ([55d1fed](https://github.com/mathysgrapotte/stimulus-py/commit/55d1fed95950adc2782a73a90690783786c9fb85) by mgrapotte).


## [0.2.4](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.2.4) - 2025-02-07

<small>[Compare with 0.2.3](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.3...0.2.4)</small>

### Bug Fixes

- test were failing, made format to fix it. ([19dce1f](https://github.com/mathysgrapotte/stimulus-py/commit/19dce1f2f6fc069a24991a843249f7edc876bb43) by mgrapotte).


## [0.2.5](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.3.0) - 2025-02-12

<small>[Compare with 0.2.4](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.4...0.3.0)</small>

### Features

- removed test results from git push. ([3f3f516](https://github.com/mathysgrapotte/stimulus-py/commit/3f3f5161e54dfc2d4db40ff59e5a2e924d30bfe4) by mgrapotte).
- added dockerfiles. ([c18489f](https://github.com/mathysgrapotte/stimulus-py/commit/c18489f59ba7e0f36c5f90a7ce51248c0bb49a07) by mgrapotte).

### Bug Fixes

- fixed duplicate in ruff config. ([eb9d3bf](https://github.com/mathysgrapotte/stimulus-py/commit/eb9d3bf566c7ab94c02f30b85d5a946b440711b9) by mgrapotte).
- update ruff config to ignore shadowing python typing. ([55d33be](https://github.com/mathysgrapotte/stimulus-py/commit/55d33be8a6068e3b4649275ec1bf3827c21ffb9a) by mgrapotte).
- renamed dockerfiles directory as it was causing ci to crash. ([6c19df4](https://github.com/mathysgrapotte/stimulus-py/commit/6c19df4589c2ff6e3a5ded53d561ba50dd4255f0) by mgrapotte).
- ray init needs to be called in the run command otherwise ray cluster is not found. ([c9526d6](https://github.com/mathysgrapotte/stimulus-py/commit/c9526d60c909ddc8d8dabd751380c5d5e16b8d44) by mgrapotte).
- ray init needs to be called in the run command, otherwise ray cluster is not found. ([4ebd495](https://github.com/mathysgrapotte/stimulus-py/commit/4ebd495ba348698252598d8e7a3575cead057593) by mgrapotte).


## [0.3.0](https://github.com/mathysgrapotte/stimulus-py/releases/tag/0.3.0) - 2025-03-18

<small>[Compare with 0.2.6](https://github.com/mathysgrapotte/stimulus-py/compare/0.2.6...0.3.0)</small>

### Features

- get device can now return mps backend objecs. ([afd4e30](https://github.com/mathysgrapotte/stimulus-py/commit/afd4e300424b457986a8c69abd324e1a68c061f7) by mathysgrapotte).
- config parser now also responsible for dealing with suggesting parameters. ([be51083](https://github.com/mathysgrapotte/stimulus-py/commit/be51083c7a98eb2a8c753aa7550fa0748323e6d6) by mgrapotte).
- now supports variable lists in search space definition. ([ff5e0b8](https://github.com/mathysgrapotte/stimulus-py/commit/ff5e0b85957eff5ea31ef0e8954382d0d7f35f3e) by mgrapotte).
- added extra test znf data. ([ab15e5e](https://github.com/mathysgrapotte/stimulus-py/commit/ab15e5ed897b9445746e2e458f822f1a2c54537d) by mgrapotte).
- added support for stimulus-split-yaml-split and stimulus-split-yaml-transform. ([d85f6e0](https://github.com/mathysgrapotte/stimulus-py/commit/d85f6e0202a88ab70adcbc67efca6c09191e98a2) by mgrapotte).

### Bug Fixes

- fixed a bug where predict was not transfering the batch to the device (when device is not CPU). ([1fa3e44](https://github.com/mathysgrapotte/stimulus-py/commit/1fa3e4408225b726358d1b52ee46970f4251b5c1) by mathysgrapotte).
- polars.vstack is deprecated, replaced by pl.concat. ([989a1fd](https://github.com/mathysgrapotte/stimulus-py/commit/989a1fd0f7c40c251bc294575c37f18b1e08f8e7) by mgrapotte).
- data handlers was calling the transform method instead of transform_all and passing a pl.series instead of a list. ([8463e90](https://github.com/mathysgrapotte/stimulus-py/commit/8463e90f9e1c5780a3f98eda9ed35cc57c5dd58c) by mgrapotte).
- tests should now use temp_paths. ([5368518](https://github.com/mathysgrapotte/stimulus-py/commit/536851889a5c5408a461b95845dc92d628eda3f2) by mathysgrapotte).
- fix issue where spearmanr would cause a warning when labels/predictions would be constant, now outputs 0.0 correlation. ([ecd9799](https://github.com/mathysgrapotte/stimulus-py/commit/ecd9799903e4cf7274cb77159a6be17799702df1) by mgrapotte).
- add option in case of divide by zero which disables warnings. ([ddef6f5](https://github.com/mathysgrapotte/stimulus-py/commit/ddef6f54aa9507dde84f92cb3e6d80e407385436) by mgrapotte).
- fix importlib import errors. ([af20925](https://github.com/mathysgrapotte/stimulus-py/commit/af209257074f6af1b75b27983ad0dcfa2a3847c7) by mgrapotte).
- added importlib_metadata dependencies. ([57d1e35](https://github.com/mathysgrapotte/stimulus-py/commit/57d1e359a84c713361f67f15da9aefc73843b62d) by mgrapotte).
- teardown is now a fixture with autouse set to true. ([414f315](https://github.com/mathysgrapotte/stimulus-py/commit/414f315d960d3c7fe09893574c93eaad70d8d80e) by mgrapotte).
- redundant teardown. ([7991a57](https://github.com/mathysgrapotte/stimulus-py/commit/7991a57067688bd388dccf055c0ddcfc617f9619) by mgrapotte).
- made teardown redundant. CI was failing because test is exectuting in parralel, CI workers do not share fixture state, so teardown was not called in all tests. ([853a0a2](https://github.com/mathysgrapotte/stimulus-py/commit/853a0a2cce7bd40d490dc796841bc4df9d30dbad) by mgrapotte).
- updated ray version since ray 2.43 introduces breaking changes. ([dd0975f](https://github.com/mathysgrapotte/stimulus-py/commit/dd0975fae5308864afadc7cd7641ab667b2a4016) by mgrapotte).
- fix ray not shutting down properly in the tests. ([6b58655](https://github.com/mathysgrapotte/stimulus-py/commit/6b5865573340b2e536e39673b46e1eeb7545e8bd) by mgrapotte).
- removed minor bugs. ([2493bfb](https://github.com/mathysgrapotte/stimulus-py/commit/2493bfbe2f6a2ba8dbb919feab39ec3e09d84588) by mgrapotte).
- now path can be given as intended to shuffle-csv. ([2fe4a0c](https://github.com/mathysgrapotte/stimulus-py/commit/2fe4a0ce1b69e046a3f1c5b68b9abd6b03363cdb) by mgrapotte).
- fix linting ([dfb7fca](https://github.com/mathysgrapotte/stimulus-py/commit/dfb7fcadf3226ec13abc4b3adc8e233752bd8d68) by mgrapotte).
- change -j flag to -y flag ([48b7ea9](https://github.com/mathysgrapotte/stimulus-py/commit/48b7ea9c6f926cc90e6183c7331355e0179ba54a) by V-E-D).
- fix linting issues. ([7c2dd2a](https://github.com/mathysgrapotte/stimulus-py/commit/7c2dd2a688d1b8f51f0c760261c9c906fdca44c7) by mgrapotte).
- running make format. ([736d6fc](https://github.com/mathysgrapotte/stimulus-py/commit/736d6fc29feb9930fd522a293394a347f5fc346c) by mgrapotte).

### Code Refactoring

- added extra yaml subconfigs. ([11d2405](https://github.com/mathysgrapotte/stimulus-py/commit/11d240589c47f588fae333066875a6c011e47318) by mgrapotte).
- added ibis test files. ([0401fac](https://github.com/mathysgrapotte/stimulus-py/commit/0401fac49585772e7f0168890f0aaafd736aa216) by mgrapotte).
- removing non-conclusive test files. ([bb5a4f7](https://github.com/mathysgrapotte/stimulus-py/commit/bb5a4f7fc6de4e2175aaa1d46065aee8ed75837e) by mgrapotte).
- udpate check_model and tests. ([59ae38a](https://github.com/mathysgrapotte/stimulus-py/commit/59ae38a6f47e871dd4613b2deb6ee125f0ff5478) by mgrapotte).
- removed test_model_yaml since it has been moved to learner/interface. ([4b5423b](https://github.com/mathysgrapotte/stimulus-py/commit/4b5423b163e8f9ab054731524fcf2780b852354f) by mgrapotte).
- changed typing to follow optuna refactor. ([eed338d](https://github.com/mathysgrapotte/stimulus-py/commit/eed338dc71042debfb99f5b4f0a023d0631cf2f5) by mgrapotte).
- tuning cli now up to date with optuna refactor. ([6d8c431](https://github.com/mathysgrapotte/stimulus-py/commit/6d8c43175f5dfe8979d7e87b545b54d81aadd54e) by mgrapotte).
- removed tune related in tests. ([a937370](https://github.com/mathysgrapotte/stimulus-py/commit/a937370231fc542e09a0c5ba30de7c81ff164431) by mgrapotte).
- removed tune related in src. ([da28b82](https://github.com/mathysgrapotte/stimulus-py/commit/da28b829b5833e661c615d65eb6a5f6419ca2cb8) by mgrapotte).
- finished optuna_tune. ([234856b](https://github.com/mathysgrapotte/stimulus-py/commit/234856b52bce60ccb101aa4e83aea863a0c60318) by mgrapotte).
- implemented optuna support, inital commit. ([173d54f](https://github.com/mathysgrapotte/stimulus-py/commit/173d54f2bc1298a75ef40fd0520f3bd5924fbf57) by mgrapotte).
- made dir paths shorter. ([4f33101](https://github.com/mathysgrapotte/stimulus-py/commit/4f331014b665c39c1532ed6b86891d528590cccd) by mgrapotte).
- tuning adapted to the new paradigm. ([7b6ed78](https://github.com/mathysgrapotte/stimulus-py/commit/7b6ed788ebef2f31445f0e8ab605435f21e3abab) by mgrapotte).
- added tuning cli refactoring. ([6a371be](https://github.com/mathysgrapotte/stimulus-py/commit/6a371bea23d5fa94987221d7d49329b989edeab0) by mgrapotte).
- refactored transform_csv to use click. ([806b9b6](https://github.com/mathysgrapotte/stimulus-py/commit/806b9b624a5a2abca3fc22925096edf325652661) by mathysgrapotte).
- Change split_transforms to the new format ([ec3784b](https://github.com/mathysgrapotte/stimulus-py/commit/ec3784b0e0cc7f58432674ec6ce3cf89947c2d9c) by itrujnara).
- added refactoring for split_csv cli. ([92558c0](https://github.com/mathysgrapotte/stimulus-py/commit/92558c07acef4fa5a6f5622ec75e08393882226f) by mathysgrapotte).
- updated shuffle_csv to the new paradigm. ([e6a0e23](https://github.com/mathysgrapotte/stimulus-py/commit/e6a0e23800776898dbb7ac669faa0c049b645ad3) by mgrapotte).
- added refactoring for check-model cli. ([09044c7](https://github.com/mathysgrapotte/stimulus-py/commit/09044c7b7b671d05e9df15d90679ce421b5b335f) by mgrapotte).
- added main cli. ([01d4532](https://github.com/mathysgrapotte/stimulus-py/commit/01d4532b4c4e2a9faee8c08004c2bba9f6b56480) by mgrapotte).
- gets a torchdataset as input. ([973e6f6](https://github.com/mathysgrapotte/stimulus-py/commit/973e6f6d1053417834d83c39c706638d8b3834d2) by mgrapotte).
- moved raytune_learner to new paradigm. ([3efda09](https://github.com/mathysgrapotte/stimulus-py/commit/3efda090ba2f6d2fbbaf41e1c563d02d89668f2d) by mgrapotte).
- fix typing accross codebase. ([ce0d5d4](https://github.com/mathysgrapotte/stimulus-py/commit/ce0d5d48d457a4d581e612c398321f7133dfd497) by mgrapotte).
- removed test_data_yaml. ([e3951f9](https://github.com/mathysgrapotte/stimulus-py/commit/e3951f9f5e3642c010e42854501810e172901f7e) by mgrapotte).
- changed import paths and removed unused fixture. ([8150042](https://github.com/mathysgrapotte/stimulus-py/commit/81500423f7be17a483a88f3c448cc48907ed9595) by mgrapotte).
- removed test_experiment. ([dcb2c80](https://github.com/mathysgrapotte/stimulus-py/commit/dcb2c80f52abac65b75dcc72075d459ffb654a1a) by mgrapotte).
- made data_handlers more explicit and added extra function to parse various configs. ([bdac5e4](https://github.com/mathysgrapotte/stimulus-py/commit/bdac5e4140354a761e7e00030d68e7e7c25abed1) by mgrapotte).
- added tests for config loading interface. ([3c4e9b4](https://github.com/mathysgrapotte/stimulus-py/commit/3c4e9b4bf71008a98fb4999df506b32ccf6172d1) by mgrapotte).
- removed duplicate files. ([21bc026](https://github.com/mathysgrapotte/stimulus-py/commit/21bc0262ec7047aea1cefad69323576d65885590) by mgrapotte).
- refactored tests for data_handlers. ([6ffcf12](https://github.com/mathysgrapotte/stimulus-py/commit/6ffcf12316d1ec5c841053f44e3557f658d49ef8) by mgrapotte).
- adapted TorchDataset to new paradigm. ([f3f13b6](https://github.com/mathysgrapotte/stimulus-py/commit/f3f13b6c40c467a646f1a0009143feb76b363567) by mgrapotte).
- removed handlertorched and happend it to data_handlers. ([377e72e](https://github.com/mathysgrapotte/stimulus-py/commit/377e72ee38609a670636e85b310cdd3c48802622) by mgrapotte).
- removed loader and manager logic. ([dbae19b](https://github.com/mathysgrapotte/stimulus-py/commit/dbae19b379664a35e6fe6e84d9d99fc7684963ca) by mgrapotte).
- added the interface module. ([90f4bb1](https://github.com/mathysgrapotte/stimulus-py/commit/90f4bb16a62ed40b92876ee0b52d664ef4914627) by mgrapotte).
- data_config interface in a single module. ([34049d0](https://github.com/mathysgrapotte/stimulus-py/commit/34049d01b204456a606fd1e9994ae44efc1f2e2a) by mgrapotte).
- added boilerplate code for instantiating encoders splitters and transforms. ([fea0b87](https://github.com/mathysgrapotte/stimulus-py/commit/fea0b87173329c1f1c396b43ed3dc94ab17e9b86) by mgrapotte).
- added parse_config function signature. ([6a6d21a](https://github.com/mathysgrapotte/stimulus-py/commit/6a6d21a82bf293983841f1f7c74db567942097bd) by mgrapotte).
- changed filenames to have more consistant naming conventions. ([576885e](https://github.com/mathysgrapotte/stimulus-py/commit/576885ebb50c854866659c07e47b9d2dfb6239ef) by mgrapotte).
- renamed to model_file_interface. ([47026f8](https://github.com/mathysgrapotte/stimulus-py/commit/47026f8f09d9257702fa9220cbef71847216b24f) by mgrapotte).
- renamed data_config_parsing to data_config_splitting. ([14daba8](https://github.com/mathysgrapotte/stimulus-py/commit/14daba833ec19ee02404d3d0bed2f9f7d63dbc1d) by mgrapotte).
- splitted yaml_data_schema and data_config_parsing. ([c1df9cf](https://github.com/mathysgrapotte/stimulus-py/commit/c1df9cfc6ae5c1c181bd323ce986c3b21c6ed1a7) by mgrapotte).
- renamed yaml_data into yaml_data_schema for consistency. ([dbca1d5](https://github.com/mathysgrapotte/stimulus-py/commit/dbca1d56000ed66d6404bac73a1e0bddc0985fbb) by mgrapotte).
- made dumper cleaner. ([8802c12](https://github.com/mathysgrapotte/stimulus-py/commit/8802c128052431c6c9994f119b08de78327d4ac8) by mgrapotte).
- removed check_yaml, redundancy with pydantic functions. ([e76d246](https://github.com/mathysgrapotte/stimulus-py/commit/e76d246d09da1dc32a760821a49505bfd7043248) by mgrapotte).
