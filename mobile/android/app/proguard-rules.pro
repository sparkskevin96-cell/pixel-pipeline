# Release builds are shrunk by R8 (Flutter adds this file automatically).
# gal -> Apache commons-imaging references desktop Java classes that Android lacks;
# gal only calls Imaging.guessFormat, which doesn't touch them.
-dontwarn java.awt.**
-dontwarn javax.imageio.**
