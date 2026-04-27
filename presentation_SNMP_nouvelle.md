# Présentation SNMP — Version alternative (basée sur « Merry SNMP.docx »)

## 1. Contexte et objectifs
- **SNMP (Simple Network Management Protocol)** : protocole applicatif pour **superviser, configurer et gérer** des équipements réseau.
- Objectif : **collecter des métriques**, détecter des incidents, et éventuellement **modifier** certains paramètres.
- Transport : généralement **UDP**.
  - **Port 161** : requêtes/réponses (manager ↔ agent)
  - **Port 162** : notifications (trap/inform)

---

## 2. Pourquoi SNMP est important
- **Vision centralisée** du réseau (santé des équipements, interfaces, CPU/RAM, températures, etc.)
- **Détection rapide** d’anomalies (lien down, surcharge, panne)
- **Historisation** et tendances (capacity planning)
- **Automatisation** : alerting, tableaux de bord, actions correctives

---

## 3. Architecture SNMP (vue claire)
### 3.1 Manager (NMS)
- Station de supervision (ex: système de monitoring) qui :
  - interroge les équipements (polling)
  - reçoit des alertes (traps/informs)

### 3.2 Agent
- Processus sur l’équipement géré (routeur, switch, serveur, onduleur…)
- Expose des variables et répond aux requêtes SNMP

### 3.3 MIB + OID
- **MIB** : base de données logique hiérarchique des paramètres gérables
- **OID** : identifiant d’un objet (ex: `1.3.6.1.2.1...`)

---

## 4. Messages et opérations SNMP (ce qu’il faut retenir)
- **GET** : lire la valeur d’un OID
- **GET-NEXT** : lire l’OID suivant (parcours de tables)
- **GET-BULK** : lire un bloc (optimisé pour grandes tables)
- **SET** : modifier un OID (configuration)
- **TRAP** : notification asynchrone (pas d’accusé réception)
- **INFORM** : notification avec accusé réception (fiabilité)

---

## 5. Schéma de fonctionnement (résumé)
1) Le **Manager** envoie une requête (GET/SET) → **Agent** (UDP 161)  
2) L’**Agent** répond → **Manager**  
3) En cas d’évènement : l’**Agent** envoie un **TRAP/INFORM** → **Manager** (UDP 162)

---

## 6. Évolution des versions SNMP (comparatif)

### 6.1 SNMPv1
- **Communauté** (community string) en **clair** (faible sécurité)
- Compteurs souvent en **32 bits** → limites sur liens rapides

### 6.2 SNMPv2c
- Sécurité : toujours basée sur la **communauté en clair**
- **GET-BULK** : collecte plus efficace
- **INFORM** : meilleure fiabilité des alertes
- Support de **64 bits** (meilleur pour haut débit)

### 6.3 SNMPv3 (la version recommandée en production)
- Sécurité renforcée : authentification + (optionnel) chiffrement
- Modèles :
  - **USM** (User-based Security Model) : utilisateurs, auth, privacy
  - **VACM** (View-based Access Control Model) : contrôle d’accès aux branches MIB
- Niveaux usuels :
  - `noAuthNoPriv`
  - `authNoPriv`
  - `authPriv`

---

## 7. Démonstration Wireshark (structure de slide)
### Expérience A — SNMPv2c (GET)
**Scénario** : lecture de `sysName` via communauté `public`  
Ce qu’on observe :
- Une trame **get-request** du manager vers l’agent
- Une trame **get-response** retour avec la valeur (ex: nom du routeur)

### Points pédagogiques à montrer
- IP source/destination
- Le port UDP (161)
- L’OID demandé (`sysName.0`)
- La communauté en clair (si décodée) → argument sécurité

---

## 8. Bonnes pratiques (à mettre en conclusion)
- Préférer **SNMPv3** (auth + chiffrement si possible)
- Restreindre l’accès :
  - filtrage IP (ACL), segmentation réseau, VPN
  - limitation des OID accessibles (VACM)
- Désactiver/éviter les communautés par défaut (`public`, `private`)
- Monitorer les traps critiques (lien down, température, alimentation…)

---

## 9. Conclusion (message clé)
- SNMP est essentiel pour la **supervision réseau** (polling + alertes).
- SNMPv1/v2c sont simples mais **peu sûrs**.
- SNMPv3 apporte la **sécurité** et le **contrôle d’accès** nécessaires à un réseau moderne.

