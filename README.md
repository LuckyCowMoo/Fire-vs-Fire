# Fire v Fire

A browser extension that detects and handles unwanted content.

## Overview

Fire v Fire is currently in development, it is currently only available for FireFox
and cannot be permanently added to Firefox yet, you must re enable it every session.

this project was developed as my 3rd year project for my computer science degree at Edge Hill University,
for detailed architecture guide, justifications and design process, please see the artefact report:
https://1drv.ms/w/c/3e7335be94f967ac/IQB7488XHN9mS7t6emHDvZ4jAfCdAYI1SlWUjTnl4B16fzg?e=8bVVBt

see a demonstration video here:
https://drive.google.com/file/d/1KiV8BoDedtRiSCtTxeWaV_HGD5Qx4AB8/view?usp=drive_link
the version seen here is already slightly outdated, the long pauses are models loading, this only
needs to happen once and can be optimised.

## Limitations

All features of Fire vs Fire are currently prototypes, with many improvements planned, while many have 
expressed great interest in this project, please consider waiting until the end of May, by which 
point a more polished and accurate experience will likely be ready, thank you for your patience.

## Disclaimer

**USE AT YOUR OWN RISK**

This software is provided "as is" without warranty of any kind, express or implied. The author accepts 
no liability for any damages, losses, or issues arising from the use of this extension, including but 
not limited to:

- False positives or false negatives in AI detection
- System performance impacts
- Registry modifications during installation
- Any consequences of blocking or flagging content

This is an experimental academic project. Users are responsible for verifying the accuracy of 
detections and using the software responsibly.

## Installation

Installation instructions:
Download this repository

Open PowerShell in the root Fire vs Fire folder, then run:

powershell -ExecutionPolicy Bypass -File .\native\install-firefox.ps1

This creates and installs the python virtual environment and all dependencies, 
it also creates the link between the Firefox browser and the Echo host V2 file, 
be aware that this involves editing two registry entries on your computer that 
create a channel for messages to be sent between this specific Firefox extension 
and the specific local server python file, i am not qualified to ensure that this 
link cannot be exploited, you can revert the registry changes by running the uninstall script:

powershell -ExecutionPolicy Bypass -File .\native\uninstall-firefox.ps1

Once you have installed the local server, add the extension temporarily to Firefox by 
visiting this address In the Firefox address bar:

about:debugging#/runtime/this-firefox

Then click the load temporary add on button and select the manifest file that is in the root
Fire vs Fire file. Then pin the extension to your taskbar for the best experience in using 
the extensions drop down in the top right corner:
  
You may need to remove any default categories and make new ones to make it work.

##
########################################################################################
########################################################################################

THIS REPOSOTORY DOES NOT CONTAIN THE MODEL WEIGHT FILES, THEY ARE TO BIG FOR GIT HUB, AND ARE HOSTED SEPERATLEY AT:

https://drive.google.com/drive/folders/1x_4QxOmiIc3xp4xFyGPpL9fF-KeWcP7a?usp=sharing

move the classifier weight files into Fire vs Fire/native/classifiers, they will now be recognised by the system

########################################################################################
########################################################################################
##


Fire vs Fire will be available to install permanently to your browser via the official Firefox extension store in the
coming weeks/months, and later to additional platforms


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
