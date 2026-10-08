# Holdout download

The plan in `protocol.json` was frozen before any holdout data was downloaded (sha256 `da91cf01f8b8369c36b531e5da917757bb409c7fbbe4f12282275dca23c97c30`). Do not open, plot or inspect these recordings before the models are locked.

Needs about 72 GB peak, or about 28 GB after deleting the zips. Source: [Facemap dataset v2](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957), CC-BY-NC 4.0.

```sh
mkdir -p ~/neuron_transformer/data/holdout && cd ~/neuron_transformer/data/holdout

# Set A: 3 later visual sessions (zip 31.8 GB; md5 must print 2ae0287ee34267aa7e2dd198a839f215)
curl -L -C - -o neural_data_visual.zip https://ndownloader.figshare.com/files/52543310
md5 neural_data_visual.zip
unzip -j neural_data_visual.zip spont_TX60_2020_10_22_1_spks.npz spont_TX61_2020_11_02_2_spks.npz spont_VR2_2020_10_26_1_spks.npz

# Set B: 5 new sensorimotor mice (zip 12.2 GB; md5 must print e00e27ee51ddab472df49a1a6d33ded7)
curl -L -C - -o neural_data_sensorimotor.zip https://ndownloader.figshare.com/files/52543184
md5 neural_data_sensorimotor.zip
unzip -j neural_data_sensorimotor.zip spont_D3_2021_11_22_2_spks.npz spont_D4_2021_11_23_2_spks.npz spont_D7_2021_11_18_2_spks.npz spont_D8_2021_11_18_3_spks.npz spont_D9_2021_11_22_2_spks.npz

# Verify size, CRC32 and spks/run format (headers only)
cd ~/neuron_transformer && python3 experiments/2026-10-07_holdout_confirmation/verify_download.py
```

`curl -C -` resumes after an interruption. Once the script prints `ALL PASS`, the zips can be deleted. `spont_D8_2021_11_29_1` is deliberately excluded.
