# This file sets up a Linux desktop entry, and moves Daedalus to the home directory.

printf "Moving the Daedalus executable and icon to ~/daedalus..."

chmod +x daedalus

if [ ! -d ~/daedalus ]; then
  mkdir ~/daedalus
fi

cp daedalus ~/daedalus
cp icon.png ~/daedalus/icon.png

# Update the desktop entry with the absolute path.
sed "s|~|$HOME|g" daedalus.desktop > ~/.local/share/applications/daedalus.desktop

printf "\nComplete. You can launch Daedalus through the GUI, eg search \"Daedalus\", and/or add to favorites.\n"