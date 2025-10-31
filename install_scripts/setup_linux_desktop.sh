# This file sets up a Linux desktop entry, and moves the application to the home directory.

NAME_UPPER="Daedalus"
NAME="daedalus"

APP_DIR="$HOME/${NAME}"
DESKTOP_PATH="$HOME/.local/share/applications/${NAME}.desktop"

chmod +x $NAME

if [ ! -d "$APP_DIR" ]; then
  mkdir "$APP_DIR"
fi

cp "$NAME" "$APP_DIR"
cp -R gemmi "$APP_DIR"
cp icon.png "$APP_DIR/icon.png"

# We create a .desktop file dynamically here; one fewer file to manage.
cat > "$DESKTOP_PATH" <<EOF
[Desktop Entry]
Name=${NAME_UPPER}
Exec=${APP_DIR}/${NAME}
Icon=${APP_DIR}/icon.png
Type=Application
Terminal=false
Categories=Development;Science;Biology;
Comment=Molecule and protein viewer
EOF

chmod +x "$DESKTOP_PATH"

# If the cuda FFT lib is packaged with the download, move it to the correct place.
cufft_lib="libcufft.so.12"
if [ -f "./$cufft_lib" ]; then
  sudo cp "./$cufft_lib" /usr/lib/
  printf "Moved the libcufft.so.12 library (for the cuFFT dependency)  to /usr/lib."
fi

printf "Moved the ${NAME_UPPER} executable and icon to ${APP_DIR}."
printf "\nYou can launch ${NAME_UPPER} through the GUI (e.g., search \"${NAME_UPPER}\") and/or add it to favorites.\n"


read -p "Install gemmi from apt, to support unprocessed electron density files? [y/n] " ans
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
  sudo apt install gemmi
  printf "\ngemmi installed. You can uninstall it with sudo apt remove gemmi.\n"
fi