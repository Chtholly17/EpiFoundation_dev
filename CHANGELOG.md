# Changelog

All notable changes to the EpiFoundation project will be documented in this file.

## [1.0.0] - 2024-10-27

### Added
- Initial release of EpiFoundation codebase
- Core training scripts:
  - `pretrain.py`: Distributed pretraining with DDP support
  - `finetune.py`: Task-specific finetuning
  - `eval.py`: Model evaluation and metrics computation
- Model architecture:
  - Transformer-based encoder for multi-omics data
  - Cross-modal attention for ATAC-RNA integration
  - Flash attention for efficient computation
  - Zero-inflated regression head for expression prediction
- Data processing:
  - Preprocessing pipeline for single-cell data
  - Custom data loaders for paired ATAC-RNA data
  - Gene/peak tokenization
- Configuration system:
  - YAML-based configuration files
  - Separate configs for pretrain/finetune/eval
  - 32 example configurations
- Documentation:
  - Comprehensive README with usage instructions
  - Quick start guide
  - Data preparation guide
  - Configuration parameter reference
  - Project structure documentation
- Utilities:
  - Training utilities (checkpointing, logging, metrics)
  - Loss functions (masked MSE, zero-inflated loss)
  - Distributed training support

### Features
- **Multi-GPU Training**: Full DDP support for efficient training
- **Mixed Precision**: Automatic mixed precision (AMP) training
- **Flexible Configuration**: YAML-based experiment configuration
- **Modular Architecture**: Easy to extend and customize
- **Comprehensive Documentation**: Detailed guides for all use cases

### Dependencies
- PyTorch >= 2.0.0
- scanpy >= 1.9.0
- anndata >= 0.8.0
- And more (see requirements.txt)

### File Structure
```
EpiFoundation_dev/
├── Core scripts (pretrain.py, finetune.py, eval.py)
├── Model architecture (model/)
├── Data processing (data/)
├── Configuration files (configs/)
├── Documentation (docs/, *.md)
├── Example scripts (scripts/)
└── Utilities (utils.py, loss/, tokenizer/)
```

### Notes
- Cleaned codebase with only essential files for training pipeline
- Removed test, cluster, and marker analysis scripts
- Focused on pretrain → finetune → eval workflow
- All documentation written in English
- Example configurations for various tasks

---

## Future Plans

### Planned Features
- [ ] Support for additional modalities (protein, spatial)
- [ ] More efficient attention mechanisms
- [ ] Automated hyperparameter tuning
- [ ] Pre-trained model zoo
- [ ] Inference API
- [ ] Docker container for easy deployment
- [ ] Benchmarking suite
- [ ] More example datasets

### Known Issues
- None currently reported

### Contributing
Contributions are welcome! Please follow the standard GitHub workflow:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

For questions or issues, please open an issue on GitHub.

