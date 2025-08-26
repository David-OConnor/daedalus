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

printf "Moved the ${NAME_UPPER} executable and icon to ${APP_DIR}."
printf "\nYou can launch ${NAME_UPPER} through the GUI (e.g., search \"${NAME_UPPER}\") and/or add it to favorites.\n"