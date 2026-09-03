function(newDoc, oldDoc, userCtx, secObj) {
    if('_admin' in userCtx.roles)
        return;
    if(!userCtx.name) {
        throw({'forbidden': 'auth first before update something'});
    }
    if(!secObj.admins.names.includes(userCtx.name)) {
        throw({'forbidden': 'user is not allowed'});
    }
}