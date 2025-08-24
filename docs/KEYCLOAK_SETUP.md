# Keycloak Setup Guide
This guide provides instructions for running and configuring a local Keycloak server for development purposes.



## 1. Running Keycloak with Docker

To start a local Keycloak instance, run the following command in your terminal. This will start Keycloak in development mode and make it accessible on port 8090.

```bash
Open: docker compose up -d
health check : curl -f http://127.0.0.1:9000/health/ready && echo "Keycloak ready ✅"

Closing: docker compose down

```

- **Access the Admin Console:** [http://localhost:8090](http://localhost:8090)
- **Admin Username:** `admin`
- **Admin Password:** `admin`

## 2. Initial Keycloak Configuration

Log in to the admin console and follow these steps to set up the environment for your application.

### a. Create a New Realm

It is best practice to isolate your application's configuration in its own realm.

1.  Hover over the "Master" realm name in the top-left corner and click **Add realm**.
2.  Enter `rag_app` as the **Name** and click **Create**.
3.  You will be automatically switched to the new `rag_app` realm.

### b. Create a Client

The "Client" in Keycloak represents your FastAPI application.

1.  In the `rag_app` realm, navigate to **Clients** from the left-hand menu.
2.  Click **Create**.
3.  Configure the client as follows:
    - **Client ID:** `fastapi-client`
    - **Client Protocol:** `openid-connect`
    - **Root URL:** Leave blank for now.
4.  Click **Save**.
5.  On the client settings page that appears, make the following changes:
    - **Access Type:** `public`
    - **Valid Redirect URIs:** `http://localhost:8000/*` (This is for your frontend application; adjust if its URL is different).
    - **Web Origins:** `+` (This allows all origins for CORS, which is convenient for development. For production, you should restrict this to your frontend's actual domain, e.g., `http://localhost:8000`).
6.  Click **Save**.

### c. Create Roles

These roles will correspond to the access levels in your application.

1.  In the `rag_app` realm, navigate to **Roles**.
2.  Click **Add Role** for each of the following roles:
    - `personal`
    - `assistance`
    - `common_rules`
    - `mine`

### d. Create a User and Assign Roles

1.  In the `rag_app` realm, navigate to **Users**.
2.  Click **Add user**.
3.  Fill in a **Username** (e.g., `testuser`) and click **Save**.
4.  Go to the **Credentials** tab for the new user.
5.  Click **Set Password**, enter a password, and turn the **Temporary** switch off. Click **Save**.
6.  Go to the **Role Mappings** tab.
7.  Use the **Assign Role** button to assign the desired roles (e.g., `personal`, `assistance`) to this user.
