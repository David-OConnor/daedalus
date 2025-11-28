//! Convert between molecules and SMILES text

use std::{collections::HashMap, io};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::{Element, Element::*};

use crate::molecule::{Atom, Bond, MoleculeCommon};

impl MoleculeCommon {
    pub fn to_smiles(&self) -> String {
        return String::new(); // todo temp

        if self.atoms.is_empty() {
            return String::new();
        }

        // Minimal DFS linearization with explicit bond symbols for non-single.
        let mut out = String::new();
        let mut visited = vec![false; self.atoms.len()];
        let mut stack: Vec<(usize, Option<usize>)> = Vec::new();

        // Start each disconnected component.
        let mut first_component = true;
        for start in 0..self.atoms.len() {
            if visited[start] {
                continue;
            }
            if !first_component {
                out.push('.');
            }
            first_component = false;

            stack.push((start, None));
            while let Some((u, parent)) = stack.pop() {
                if visited[u] {
                    continue;
                }
                visited[u] = true;

                if let Some(p) = parent {
                    // Print bond symbol if non-single.
                    if let Some(b) = self.find_bond(p, u) {
                        match b.bond_type {
                            BondType::Single => {} // implicit
                            BondType::Double => out.push('='),
                            BondType::Triple => out.push('#'),
                            _ => {} // keep minimal
                        }
                    }
                }

                out.push_str(&self.atoms[u].element.to_letter());

                // Push children; simple heuristic: neighbors except parent.
                // We add parentheses for branch degree > 1.
                let nbrs: Vec<usize> = self.adjacency_list[u]
                    .iter()
                    .copied()
                    .filter(|&v| !visited[v])
                    .collect();

                if !nbrs.is_empty() {
                    // First neighbor continues main chain; others as branches.
                    let mut it = nbrs.into_iter();
                    if let Some(first) = it.next() {
                        stack.push((first, Some(u)));
                    }
                    for v in it {
                        out.push('(');
                        // Print bond symbol for branched bond if needed.
                        if let Some(b) = self.find_bond(u, v) {
                            match b.bond_type {
                                BondType::Single => {}
                                BondType::Double => out.push('='),
                                BondType::Triple => out.push('#'),
                                _ => {}
                            }
                        }
                        out.push_str(&self.atoms[v].element.to_letter());
                        // Continue along one neighbor if it exists; otherwise close.
                        // For minimal output, just close immediately; deeper branches
                        // will be reached when that neighbor is popped from stack.
                        out.push(')');
                        stack.push((v, Some(u)));
                    }
                }
            }
        }

        out
    }

    pub fn from_smiles(data: &str) -> io::Result<Self> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut bonds: Vec<Bond> = Vec::new();
        let mut adjacency_list: Vec<Vec<usize>> = Vec::new();
        let mut atom_posits: Vec<Vec3> = Vec::new();

        let mut current: Option<usize> = None;
        let mut last_bond: Option<BondType> = None;
        let mut branch_stack: Vec<usize> = Vec::new();
        let mut ring_map: HashMap<u8, (usize, BondType)> = HashMap::new();

        let mut chars = data.chars().peekable();
        let mut next_serial: u32 = 1;

        while let Some(ch) = chars.peek().copied() {
            match ch {
                // Bonds
                '-' => {
                    last_bond = Some(BondType::Single);
                    chars.next();
                }
                '=' => {
                    last_bond = Some(BondType::Double);
                    chars.next();
                }
                '#' => {
                    last_bond = Some(BondType::Triple);
                    chars.next();
                }

                // Branching
                '(' => {
                    if let Some(ci) = current {
                        branch_stack.push(ci);
                    }
                    chars.next();
                }
                ')' => {
                    current = branch_stack.pop();
                    chars.next();
                }

                // Disconnected components
                '.' => {
                    current = None;
                    last_bond = None;
                    chars.next();
                }

                // Ring closures 0-9
                '0'..='9' => {
                    let d = ch as u8 - b'0';
                    let cur_idx = match current {
                        Some(i) => i,
                        None => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "ring digit without current atom",
                            ));
                        }
                    };
                    let bt = last_bond.unwrap_or(BondType::Single);
                    if let Some((other, bt_saved)) = ring_map.remove(&d) {
                        add_bond(
                            cur_idx,
                            other,
                            bt_saved,
                            &mut bonds,
                            &mut adjacency_list,
                            &atoms,
                        );
                    } else {
                        ring_map.insert(d, (cur_idx, bt));
                    }
                    last_bond = None;
                    chars.next();
                }

                // Atoms (organic subset, including two-letter halogens)
                _ => {
                    if let Some((sym, aromatic)) = parse_atom_token(&mut chars)? {
                        let element = Element::from_letter(sym)?;
                        let idx = atoms.len();
                        atoms.push(Atom {
                            serial_number: next_serial,
                            posit: Vec3::new(0.0, 0.0, 0.0),
                            element,
                            type_in_res: None,
                            type_in_res_general: None,
                            force_field_type: None,
                            role: None,
                            residue: None,
                            chain: None,
                            hetero: false,
                            occupancy: None,
                            partial_charge: None,
                            alt_conformation_id: None,
                        });
                        adjacency_list.push(Vec::new());
                        atom_posits.push(Vec3::new(0.0, 0.0, 0.0));

                        if let Some(prev) = current {
                            let bt = last_bond.take().unwrap_or_else(|| {
                                // minimal behavior: treat implicit/aromatic bonds as single
                                BondType::Single
                            });
                            add_bond(prev, idx, bt, &mut bonds, &mut adjacency_list, &atoms);
                        }

                        current = Some(idx);
                        next_serial += 1;
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unable to parse atom token",
                        ));
                    }
                }
            }
        }

        if !ring_map.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unclosed ring index",
            ));
        }

        Ok(MoleculeCommon {
            ident: data.trim().to_string(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata: HashMap::new(),
            visible: true,
            path: None,
            selected_for_md: false,
            entity_i_range: None,
        })
    }

    fn find_bond(&self, a: usize, b: usize) -> Option<&Bond> {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.bonds
            .iter()
            .find(|bb| (bb.atom_0 == lo && bb.atom_1 == hi) || (bb.atom_0 == hi && bb.atom_1 == lo))
    }
}

fn add_bond(
    a: usize,
    b: usize,
    bond_type: BondType,
    bonds: &mut Vec<Bond>,
    adj: &mut Vec<Vec<usize>>,
    atoms: &Vec<Atom>,
) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    bonds.push(Bond {
        bond_type,
        atom_0_sn: atoms[lo].serial_number,
        atom_1_sn: atoms[hi].serial_number,
        atom_0: lo,
        atom_1: hi,
        is_backbone: false,
    });
    adj[a].push(b);
    adj[b].push(a);
}

fn parse_atom_token(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> io::Result<Option<(&'static str, bool)>> {
    // Organic subset + halogens; minimal bracket handling (rejects bracketed forms)
    let ch = match chars.peek().copied() {
        Some(c) => c,
        None => return Ok(None),
    };

    // Two-letter halogens need special-casing
    if ch == 'C' {
        // Could be "Cl"
        let mut tmp = chars.clone();
        tmp.next();
        if let Some('l') = tmp.peek().copied() {
            // Cl
            chars.next(); // 'C'
            chars.next(); // 'l'
            return Ok(Some(("Cl", false)));
        } else {
            chars.next();
            return Ok(Some(("C", false)));
        }
    }
    if ch == 'B' {
        let mut tmp = chars.clone();
        tmp.next();
        if let Some('r') = tmp.peek().copied() {
            chars.next(); // 'B'
            chars.next(); // 'r'
            return Ok(Some(("Br", false)));
        } else {
            chars.next();
            return Ok(Some(("B", false)));
        }
    }

    // Single-letter elements / aromatic lower-case subset
    match ch {
        'c' => {
            chars.next();
            Ok(Some(("C", true)))
        }
        'n' => {
            chars.next();
            Ok(Some(("N", true)))
        }
        'o' => {
            chars.next();
            Ok(Some(("O", true)))
        }
        'p' => {
            chars.next();
            Ok(Some(("P", true)))
        }
        's' => {
            chars.next();
            Ok(Some(("S", true)))
        }
        'C' | 'N' | 'O' | 'P' | 'S' | 'F' | 'I' => {
            chars.next();
            Ok(Some((&*ch.to_string().leak(), false)))
        }
        '[' => {
            // Minimal parser: reject bracketed atoms to keep scope small for now.
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bracketed atoms not supported in minimal parser",
            ))
        }
        _ => Ok(None),
    }
}
