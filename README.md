# Fire v Fire

A browser extension that detects and handles unwanted content.

## Overview

Fire v Fire is currently in early development

this project is a continuation of my undergraduate comp sci dissertation for those who care, 
please see my full report
https://1drv.ms/w/c/3e7335be94f967ac/IQB7488XHN9mS7t6emHDvZ4jAfCdAYI1SlWUjTnl4B16fzg?e=8bVVBt

see a demonstration video here:
https://drive.google.com/file/d/1KiV8BoDedtRiSCtTxeWaV_HGD5Qx4AB8/view?usp=drive_link
the version seen here is quite outdated, accuracy, speed, UI, accessibility , and more have been 
greatly improved since

## Limitations

All features of Fire vs Fire remain r=early development, furthermore, this project and its classifiers 
are designed to work under extreme limitations, classifiers must fit on low end consumer hardware and must
run at reasonable speeds to keep up with user browsing, the accuracy of the current provided classifiers
is low compared to single target classification tools available, the benefits of this system are its 
100% hands free operation, privacy, and adaptability.

## Disclaimer

**USE AT YOUR OWN RISK**

This software is provided "as is" without warranty of any kind, express or implied. The author accepts 
no liability for any damages, losses, or issues arising from the use of this extension, including but 
not limited to:

- False positives or false negatives in AI detection
- System performance impacts
- Any consequences of blocking or flagging content

This is an experimental academic project. Users are responsible for verifying the accuracy of 
detection and using the software responsibly.

## Installation

this browser extension will not be able to classify content without first installing the accompanying native
host (a server) onto your own device, these two systems communicate in order to automatically grab content from
your webpages, determine their origin, and then alter the webpage without you input, no data ever leaves
your device.

After installing this extension via official browser stores, follow the on screen instructions

you will need to download, unzip, and run the install script for the native host, a browser extension
does not have the permissions to activate these steps for you.

## License

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

Copyright (c) 2026 Leah (Luke) Armstrong (LuckyCowMoo)

You must first acquire my express permission to use this code as a component of any project
that will make money.

See [LICENSE](LICENSE) for full terms.

## Credits

- Project art by Beti Meredith Gray (certified human)
- PyTorch and torchvision for deep learning framework
- ResNet-50 architecture from Microsoft Research
- ConvNeXt architecture from Meta AI Research
