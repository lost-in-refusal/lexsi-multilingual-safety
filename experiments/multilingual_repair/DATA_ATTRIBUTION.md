# Data attribution and modifications

Task data is derived from the human-authored **Aya Dataset**, by Singh et al.
and the Aya contributors, distributed by Cohere For AI under Apache-2.0.
The original dataset card is preserved in `licenses/Aya-dataset-card.md` from
revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` of
[CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset/tree/f9ea04583f02a8f86404ff6c58bf75fe637df8a2).
A copy of Apache-2.0 is included in `licenses/Apache-2.0.txt`.

Safety data is derived from **MultiJail**, by Yue Deng, Wenxuan Zhang,
Sinno Jialin Pan, and Lidong Bing. The source is
[DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail/tree/93d65c778925d55973b9ff30c7bccc8545494eb6),
revision `93d65c778925d55973b9ff30c7bccc8545494eb6`.
The MIT license and original copyright notice from the
[upstream repository](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/blob/main/LICENSE)
are included in `licenses/MultiJail-MIT.txt`.

The prepared JSONL files are modified derivatives: they select three languages,
filter by length and a documented lexical screen, normalize and deduplicate
requests, create new experimental partitions and identifiers, add system prompts,
and encode model tokens and loss masks. Aya target answers retain their source
text. The preparation manifest records source hashes, transformation settings,
exclusion counts, and output hashes. These modifications are not an endorsement
by the source authors. Model-generated responses and review annotations are
separate experimental outputs, not original dataset labels.

Model weights and third-party tool repositories are not redistributed in this
package. Reproduction downloads them from the pinned original sources under
their respective conditions. The data licenses above apply to their stated
data sources; they do not relicense the base model or the Lexsi toolkits.
